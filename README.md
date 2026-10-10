# gpu-lbfgs

Batched L-BFGS structure relaxation on the GPU. Many structures are relaxed at once: at every
step all unconverged structures go through the potential in one batched call, and the L-BFGS
update is done for all of them together with PyTorch tensors.

The optimizer reproduces `ase.optimize.LBFGS` (no line search), either on
`ase.filters.FrechetCellFilter` (positions and cell, zero pressure) or on the atoms alone
(positions only). All settings default to ASE's values except `alpha`, which is 10 here instead of
ASE's 70 (see [Choosing alpha](#choosing-alpha)); with `alpha=70` the results are identical to ASE's
defaults. It works with any potential that can evaluate a list of structures. A wrapper for ORB
models and one for any ASE calculator are included.

## Install

```bash
git clone <this repository>
cd gpu-lbfgs
pip install -e .              # numpy, torch, ase>=3.23
pip install -e ".[orb]"       # optional: orb-models, for orb_predict
```

## Quick start

With an ORB model:

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator
from gpu_lbfgs import relax, orb_predict

model, adapter = pretrained.orb_v3_conservative_inf_mpa(device="cuda")
predict = orb_predict(ORBCalculator(model, adapter, device="cuda"))

structures = read("structures.extxyz", index=":")
results = relax(structures, predict, fmax=0.02, steps=500, batch_size=256)

for r in results:
    print(r.converged, r.steps, r.energy / len(r.atoms))
relaxed = [r.atoms for r in results]
```

With any ASE calculator (evaluated structure by structure, so mainly for testing):

```python
from ase.calculators.emt import EMT
from gpu_lbfgs import relax, ase_calculator_predict

results = relax(structures, ase_calculator_predict(EMT()), fmax=0.01, device="cpu")
```

A command-line example is in `examples/relax_with_orb.py`:

```bash
python examples/relax_with_orb.py structures.extxyz --fmax 0.02 --out relaxed.extxyz
```

## API

### `relax(atoms_list, predict, *, fmax=0.05, steps=500, relax_cell=True, device=None, batch_size=None, maxstep=0.2, memory=100, alpha=10.0, damping=1.0, callback=None)`

| Argument | Meaning |
|---|---|
| `atoms_list` | list of `ase.Atoms`; they are not modified |
| `predict` | batched potential, see below |
| `fmax` | stop a structure when its largest force is below `fmax` (eV/A). With `relax_cell=True` this is the largest generalized force of `FrechetCellFilter`, as in ASE |
| `steps` | largest number of L-BFGS steps per structure |
| `relax_cell` | `True`: positions and cell at zero pressure. `False`: positions only |
| `device` | where the optimizer runs (default `cuda` if available) |
| `batch_size` | structures per batch (default: all at once); batches are processed one after another |
| `maxstep`, `memory`, `damping` | as in `ase.optimize.LBFGS`, same defaults (0.2 A, 100, 1) |
| `alpha` | initial inverse Hessian is 1/`alpha` (eV/A^2). Default 10; ASE's default is 70 |
| `callback` | `callback(step, active, energies)` after each evaluation: indices of the structures still running and their energies |

Returns one `RelaxResult` per structure, in input order:

| Field | Meaning |
|---|---|
| `atoms` | relaxed structure (same atom order, cell in the input frame) with a `SinglePointCalculator` holding the final energy, forces and stress; `None` if the relaxation failed |
| `converged` | the force criterion was reached |
| `steps` | L-BFGS steps taken |
| `energy`, `energy_initial` | total energy (eV) at the end and at the start |
| `fmax` | final largest (generalized) force (eV/A) |
| `error` | message if the relaxation failed |

### The `predict` function

`predict(atoms_list)` evaluates a list of `ase.Atoms` and returns a dict:

| Key | Shape | Unit |
|---|---|---|
| `energy` | `[B]` | eV (total energy per structure) |
| `forces` | `[total number of atoms, 3]`, structures concatenated in order | eV/A |
| `stress` | `[B, 6]` (ASE Voigt order xx, yy, zz, yz, xz, xy) or `[B, 3, 3]`; only used if `relax_cell=True` | eV/A^3, ASE sign convention |

Values can be torch tensors on any device or NumPy arrays. `predict` is called outside
`torch.no_grad()`, so potentials that compute forces with autograd work. To use another
machine-learning potential, write this function around its batched inference and keep the
results on the GPU. Ready-made functions:

- `orb_predict(orb_calculator)`: one ORB call per batch. For conservative models it uses the
  energy gradients (`grad_forces`, `grad_stress`), otherwise the direct heads, the same choice
  as `ORBCalculator` itself.
- `ase_calculator_predict(calculator, stress=True)`: any ASE calculator, one structure at a time.

## How it works

Each structure is described by the coordinates of `FrechetCellFilter`: with F the deformation of
the cell relative to the input cell (cell = cell0 F^T),

- positions part: r F^-T (r: Cartesian positions),
- cell part: N log(F), with N the number of atoms (ASE's default `exp_cell_factor`).

The matching generalized forces are computed from the potential's forces and stress. The cell
part needs the adjoint of the derivative of the matrix exponential; it is obtained from one 6x6
block matrix exponential per structure.

The L-BFGS update is that of `ase.optimize.LBFGS.step` without line search: two-loop recursion
with initial inverse Hessian 1/alpha, the last `memory` pairs (s, y), and the step scaled down so
that its longest row is at most `maxstep`. Structures of different sizes are padded with zero
rows, which do not change any inner product or step length. A structure leaves the batch when
it converges or reaches `steps`, so later calls of `predict` only contain the structures still
being relaxed. The optimizer state is kept in float64.

Structures are passed to `predict` as `ase.Atoms`, so graph construction is whatever the
potential's own batching does (for ORB, `adapter.from_ase_atoms_list`).

## Choosing alpha

`alpha` sets the initial inverse Hessian 1/alpha, i.e. how large the first steps are before
L-BFGS has learned the curvature (every step is still capped at `maxstep`). The default here is
`alpha=10`; ASE uses 70. In an earlier test (455 corrupted MP-20 structures relaxed with ORB v3 on
the GPU, `maxstep=0.2`), a smaller alpha made typical structures converge in fewer steps, but a few
more structures ended in higher-energy local minima:

| | alpha = 70 (ASE) | alpha = 10 (default here) |
|---|---|---|
| converged after 30 steps (energy change < 1e-3 eV/atom) | 84% | 91% |
| converged after 50 steps | 99% | 99.6% |
| median energy above hull after 20 steps (eV/atom) | 0.015 | 0.009 |
| mean energy above hull after 300 steps (eV/atom) | 0.015 | 0.028 |
| energy above hull > 0.1 eV/atom after 300 steps | 1.3% | 3.3% |

Each step costs the same, so the speed difference is the number of steps. Use `alpha=70` to
reproduce ASE's defaults exactly, or if fewer high-energy end points matter more than speed.
Do not lower `maxstep` to make relaxations "safer": with `maxstep=0.05` the same structures needed
many more steps.

## Tests

`tests/test_against_ase.py` compares `relax` with `ase.optimize.LBFGS` using ASE's EMT potential
on the CPU (five strained, sheared and rattled metal cells of 4 to 32 atoms, relaxed together):

- positions and cell (`FrechetCellFilter`) and positions only, both with the default alpha (10,
  against ASE run with `alpha=10`) and with `alpha=70` (ASE's defaults): the same number of steps
  for every structure, final energies equal to 1e-8 eV, positions and cell to 1e-6 A;
- relaxing all structures in one batch or one at a time gives the same results;
- the step limit, the callback, and that the inputs are left unchanged.

```bash
python -m pytest tests          # or, without pytest: python tests/test_against_ase.py
```

## Benchmark

`examples/benchmark_orb.py` on one NVIDIA A100 80GB PCIe with `orb_v3_conservative_inf_mpa`
(precision `float32-high`): 512 MP-20 test structures (5,232 atoms) with a random symmetric strain
(standard deviation 3%) and 0.1 A rattle, relaxed with positions and cell to fmax = 0.02 eV/A
(at most 500 steps). Reference: `ase.optimize.LBFGS(FrechetCellFilter(atoms))` with
`ORBCalculator`, one structure at a time, on the first 64 structures. These numbers were measured
with `alpha=70` for both methods (the default before it was changed to 10); use
`--alpha` to choose.

| Method | Structures per second | Speed-up |
|---|---|---|
| ASE L-BFGS, one structure at a time | 1.14 | 1x |
| gpu_lbfgs, batch_size 32 | 7.35 | 6.4x |
| gpu_lbfgs, batch_size 128 | 20.4 | 17.9x |
| gpu_lbfgs, batch_size 512 | 35.0 | 30.8x |

All 512 structures converged (38.8 steps on average). Final energies of the 64 structures relaxed
both ways: median difference 0.08 meV/atom, largest 0.66 meV/atom. ORB on the GPU is not
bit-reproducible, so the two runs follow slightly different trajectories and usually end after a
different number of steps; with a deterministic potential they are identical (see Tests).

## Notes

- Zero external pressure only; the full cell is relaxed (no constraints or masks).
- For ORB on the GPU, results are not bit-reproducible run to run, so trajectories can differ
  slightly from a serial ASE run; with deterministic potentials (the EMT test) they are identical.
- With direct (non-conservative) force models, the forces are not exactly the negative gradient of
  the energy; L-BFGS then follows the forces, as ASE does.
