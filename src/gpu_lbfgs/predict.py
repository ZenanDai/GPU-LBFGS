"""Ready-made predict functions for gpu_lbfgs.relax.

A predict function takes a list of ase.Atoms and returns
    {"energy": [B] (eV), "forces": [sum of atoms, 3] (eV/A), "stress": [B, 6] or [B, 3, 3] (eV/A^3)}
as torch tensors or NumPy arrays. For another machine-learning potential, write the same kind of
function around its batched inference; keep the results on the GPU to avoid copies.
"""
import numpy as np


def ase_calculator_predict(calculator, stress=True):
    """Wrap any ASE calculator (evaluates structure by structure, so it is not fast).

    Useful for testing and for potentials without batched inference. Use stress=False for
    positions-only relaxations of non-periodic systems.
    """
    def predict(atoms_list):
        energy, forces, stresses = [], [], []
        for atoms in atoms_list:
            atoms.calc = calculator
            energy.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
            stresses.append(atoms.get_stress(voigt=True) if stress else np.zeros(6))
        return {"energy": np.array(energy), "forces": np.concatenate(forces), "stress": np.array(stresses)}
    return predict


def orb_predict(calculator):
    """Batched predict for an orb_models ORBCalculator: one ORB call per batch, results stay on its device.

    Uses the same outputs as ORBCalculator itself: for conservative models the energy gradients
    (grad_forces, grad_stress), otherwise the direct force and stress heads.

        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        model, adapter = pretrained.orb_v3_conservative_inf_mpa(device="cuda")
        predict = orb_predict(ORBCalculator(model, adapter, device="cuda"))
    """
    model = calculator.model
    energy_key = getattr(model, "energy_name", "energy")
    if getattr(calculator, "conservative", False):
        forces_key = getattr(model, "grad_forces_name", None)
        stress_key = getattr(model, "grad_stress_name", None)
    else:
        forces_key = getattr(model, "forces_name", "forces")
        stress_key = getattr(model, "stress_name", "stress")
    if None in (energy_key, forces_key, stress_key):
        raise ValueError("This ORB model does not expose energy, force and stress outputs")

    def predict(atoms_list):
        if getattr(calculator, "expects_charge_and_spin", False) and any(
                "charge" not in a.info or "spin" not in a.info for a in atoms_list):
            raise ValueError("This ORB model needs atoms.info['charge'] and atoms.info['spin']")
        batch = calculator.adapter.from_ase_atoms_list(
            atoms_list, max_num_neighbors=getattr(calculator, "max_num_neighbors", None),
            edge_method=getattr(calculator, "edge_method", None), device=calculator.device)
        out = calculator.model.predict(batch, split=False)
        return {"energy": out[energy_key].detach().reshape(-1), "forces": out[forces_key].detach(),
                "stress": out[stress_key].detach()}
    return predict
