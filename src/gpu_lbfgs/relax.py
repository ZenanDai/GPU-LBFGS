"""Batched structure relaxation: all structures of a batch take their L-BFGS steps together."""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .cell_filter import generalized_forces
from .lbfgs import BatchLBFGS


@dataclass
class RelaxResult:
    """Outcome for one structure.

    atoms: relaxed structure (atom order kept, cell in the input frame) with a SinglePointCalculator
        holding the final energy, forces and stress; None if the relaxation failed.
    converged: the largest (generalized) force fell below fmax.
    steps: number of L-BFGS steps taken.
    energy, energy_initial: total energy (eV) at the end and at the start.
    fmax: largest (generalized) force at the end (eV/A).
    error: message if the relaxation failed (for example non-finite model output).
    """
    atoms: object
    converged: bool
    steps: int
    energy: Optional[float]
    energy_initial: Optional[float]
    fmax: Optional[float]
    error: Optional[str] = None


def _as_tensor(value, device):
    return torch.as_tensor(value).detach().to(device=device, dtype=torch.float64)


def relax(atoms_list, predict: Callable, *, fmax=0.05, steps=500, relax_cell=True, device=None,
          batch_size=None, maxstep=0.2, memory=100, alpha=10.0, damping=1.0, callback=None):
    """Relax a list of ase.Atoms with L-BFGS, all structures of a batch at once.

    predict(atoms_list) -> dict with
        energy: [B] total energies (eV)
        forces: [sum of atoms, 3] forces (eV/A), structures concatenated in order
        stress: [B, 6] (ASE Voigt order) or [B, 3, 3] stress (eV/A^3); only needed if relax_cell
    as torch tensors (any device) or NumPy arrays. See gpu_lbfgs.predict for ready-made ones.
    predict is called outside torch.no_grad(), so models that take forces as gradients work.

    relax_cell=True relaxes positions and cell at zero pressure, like
    ase.optimize.LBFGS(ase.filters.FrechetCellFilter(atoms)); relax_cell=False relaxes positions
    only, like ase.optimize.LBFGS(atoms). fmax, steps, maxstep, memory, alpha and damping have the
    meaning of the ASE arguments (steps = largest number of L-BFGS steps). The default alpha is 10,
    not ASE's 70 (larger first steps, usually fewer steps); pass alpha=70 to reproduce ASE's defaults.
    batch_size: structures per batch (default: all at once). callback(step, active, energies) is
    called after every model evaluation with the indices (into atoms_list) of the structures still
    being relaxed and their energies (eV).
    The input structures are not modified. Returns one RelaxResult per structure, in input order.
    """
    if steps < 0 or not fmax > 0:
        raise ValueError("steps must be >= 0 and fmax > 0")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    size = batch_size or max(len(atoms_list), 1)
    for start in range(0, len(atoms_list), size):
        part = atoms_list[start:start + size]
        hook = None if callback is None else (
            lambda step, active, energies, _s=start: callback(step, [_s + i for i in active], energies))
        results += _relax_batch(part, predict, fmax=fmax, steps=steps, relax_cell=relax_cell, device=device,
                                optimizer=BatchLBFGS(maxstep, memory, alpha, damping), callback=hook)
    return results


def _relax_batch(atoms_list, predict, *, fmax, steps, relax_cell, device, optimizer, callback):
    from ase.calculators.singlepoint import SinglePointCalculator

    results = [None] * len(atoms_list)
    frames, active = [], []
    for i, atoms in enumerate(atoms_list):
        frame = atoms.copy()
        frame.calc = None
        cell = np.asarray(frame.cell)
        if not len(frame) or not np.isfinite(frame.positions).all():
            results[i] = RelaxResult(None, False, 0, None, None, None, "empty or non-finite structure")
        elif relax_cell and (not np.isfinite(cell).all() or np.linalg.det(cell) <= 0):
            results[i] = RelaxResult(None, False, 0, None, None, None, "cell relaxation needs a right-handed 3D cell")
        else:
            frames.append(frame)
            active.append(i)
    if not active:
        return results

    sizes = np.array([len(a) for a in frames])
    width = int(sizes.max())
    padded = np.zeros((len(frames), width, 3))
    for j, a in enumerate(frames):
        padded[j, :len(a)] = a.positions
    q = torch.as_tensor(padded, dtype=torch.float64, device=device)        # positions in the initial frame
    counts = torch.as_tensor(sizes, dtype=torch.float64, device=device)
    if relax_cell:
        cell0 = torch.as_tensor(np.array([a.cell.array for a in frames]), dtype=torch.float64, device=device)
        logf = torch.zeros_like(cell0)
        deformation = torch.eye(3, dtype=torch.float64, device=device).expand(len(frames), 3, 3).clone()
    initial = {}

    def finish(j, step, converged, energy, force_rows, stress, residual, error=None):
        i = active[j]
        if error is not None:
            results[i] = RelaxResult(None, False, step, None, initial.get(i), None, error)
            return
        atoms = frames[j].copy()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=force_rows,
                                           **({"stress": stress} if stress is not None else {}))
        results[i] = RelaxResult(atoms, converged, step, energy, initial[i], residual)

    for step in range(steps + 1):
        with torch.no_grad():
            positions = q @ deformation.transpose(-1, -2) if relax_cell else q
            geometry = positions.cpu().numpy()
            cells = (cell0 @ deformation.transpose(-1, -2)).cpu().numpy() if relax_cell else None
        for j, a in enumerate(frames):
            if relax_cell:
                a.set_cell(cells[j], scale_atoms=False)
            a.set_positions(geometry[j, :sizes[j]])
        try:
            out = predict(frames)              # outside no_grad: forces may come from autograd
            with torch.no_grad():
                energy = _as_tensor(out["energy"], device).reshape(-1)
                forces = _as_tensor(out["forces"], device)
                if energy.shape != (len(frames),) or forces.shape != (int(sizes.sum()), 3):
                    raise ValueError(f"predict returned energy {tuple(energy.shape)} and forces "
                                     f"{tuple(forces.shape)} for {len(frames)} structures / {int(sizes.sum())} atoms")
                row = torch.as_tensor(np.repeat(np.arange(len(frames)), sizes), device=device)
                slot = torch.as_tensor(np.concatenate([np.arange(n) for n in sizes]), device=device)
                padded_force = torch.zeros_like(q)
                padded_force[row, slot] = forces
                stress = None
                if relax_cell:
                    stress = _as_tensor(out["stress"], device)
                    if stress.shape not in ((len(frames), 6), (len(frames), 3, 3)):
                        raise ValueError(f"predict returned stress {tuple(stress.shape)}")
                    cell = cell0 @ deformation.transpose(-1, -2)
                    atom_force, cell_force, residual = generalized_forces(
                        logf, deformation, cell, padded_force, stress, counts)
                    x = torch.cat((q, logf * counts[:, None, None]), dim=1)
                    f = torch.cat((atom_force, cell_force), dim=1)
                else:
                    residual = torch.linalg.vector_norm(padded_force, dim=-1).amax(dim=-1)
                    x, f = q, padded_force
                finite = (torch.isfinite(energy) & torch.isfinite(residual)).cpu().numpy()
                energy_np, residual_np = energy.cpu().numpy(), residual.cpu().numpy()
        except Exception as exc:
            for j in range(len(frames)):
                finish(j, step, False, None, None, None, None, f"{type(exc).__name__}: {exc}")
            return results
        for j, i in enumerate(active):
            initial.setdefault(i, float(energy_np[j]))
        if callback is not None:
            callback(step, list(active), energy_np.copy())

        keep = []
        for j in range(len(frames)):
            if not finite[j]:
                finish(j, step, False, None, None, None, None, "non-finite energy or forces")
            elif residual_np[j] < fmax or step == steps:
                stress_j = None
                if stress is not None:
                    s = stress[j].cpu().numpy()
                    stress_j = s if s.shape == (6,) else s[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
                finish(j, step, bool(residual_np[j] < fmax), float(energy_np[j]),
                       padded_force[j, :sizes[j]].cpu().numpy(), stress_j, float(residual_np[j]))
            else:
                keep.append(j)
        if not keep:
            break
        if len(keep) != len(frames):
            index = torch.tensor(keep, device=device)
            pick = lambda t: t.index_select(0, index)
            q, counts, x, f = pick(q), pick(counts), pick(x), pick(f)
            if relax_cell:
                cell0, logf, deformation = pick(cell0), pick(logf), pick(deformation)
            optimizer.keep(index)
            frames, active, sizes = [frames[j] for j in keep], [active[j] for j in keep], sizes[keep]
        x = optimizer.step(x, f)
        with torch.no_grad():
            q = x[:, :width]
            if relax_cell:
                logf = x[:, width:] / counts[:, None, None]
                deformation = torch.linalg.matrix_exp(logf)
    return results
