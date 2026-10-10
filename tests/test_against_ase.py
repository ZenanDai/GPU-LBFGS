"""gpu_lbfgs.relax against ase.optimize.LBFGS with ASE's EMT potential (CPU, no GPU or ML model needed).

    python -m pytest tests      (or: python tests/test_against_ase.py)
"""
import numpy as np

try:
    import pytest
except ImportError:          # the tests also run without pytest: python tests/test_against_ase.py
    pytest = None
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

from gpu_lbfgs import ase_calculator_predict, relax

FMAX, STEPS = 0.01, 300


def structures():
    """Strained and rattled EMT metals of different sizes (exercises the padding of a batch)."""
    out = []
    for i, (symbol, rep, strain) in enumerate([("Cu", (2, 2, 2), 1.03), ("Al", (1, 1, 1), 0.97),
                                               ("Ni", (2, 1, 1), 1.02), ("Ag", (1, 2, 1), 1.04),
                                               ("Pt", (1, 1, 2), 0.98)]):
        atoms = bulk(symbol, cubic=True).repeat(rep)
        cell = np.asarray(atoms.cell) * strain
        cell[0, 1] += 0.05 * (i + 1)               # also shear the cell
        atoms.set_cell(cell, scale_atoms=True)
        atoms.rattle(0.05, seed=i)
        out.append(atoms)
    return out


def ase_reference(atoms, relax_cell):
    atoms = atoms.copy()
    atoms.calc = EMT()
    opt = LBFGS(FrechetCellFilter(atoms) if relax_cell else atoms, logfile=None)
    opt.run(fmax=FMAX, steps=STEPS)
    return atoms, opt.nsteps


def check_matches_ase_lbfgs(relax_cell):
    inputs = structures()
    results = relax(inputs, ase_calculator_predict(EMT()), fmax=FMAX, steps=STEPS,
                    relax_cell=relax_cell, device="cpu")
    for atoms, r in zip(inputs, results):
        ref, nsteps = ase_reference(atoms, relax_cell)
        assert r.error is None and r.converged
        assert r.steps == nsteps
        assert abs(r.energy - ref.get_potential_energy()) < 1e-8
        assert np.abs(r.atoms.positions - ref.positions).max() < 1e-6
        assert np.abs(np.asarray(r.atoms.cell) - np.asarray(ref.cell)).max() < 1e-6
        assert r.fmax < FMAX
        assert np.allclose(r.atoms.get_potential_energy(), r.energy)


def test_matches_ase_lbfgs_cell():
    check_matches_ase_lbfgs(True)


def test_matches_ase_lbfgs_positions_only():
    check_matches_ase_lbfgs(False)


def test_batching_does_not_change_results():
    inputs = structures()
    predict = ase_calculator_predict(EMT())
    together = relax(inputs, predict, fmax=FMAX, steps=STEPS, device="cpu")
    one_by_one = relax(inputs, predict, fmax=FMAX, steps=STEPS, device="cpu", batch_size=1)
    for a, b in zip(together, one_by_one):
        assert a.steps == b.steps
        assert abs(a.energy - b.energy) < 1e-10
        assert np.abs(a.atoms.positions - b.atoms.positions).max() < 1e-9


def test_step_limit_and_inputs_untouched():
    inputs = structures()
    before = [a.positions.copy() for a in inputs]
    seen = []
    results = relax(inputs, ase_calculator_predict(EMT()), fmax=1e-6, steps=3, device="cpu",
                    callback=lambda step, active, energies: seen.append((step, list(active))))
    assert all(not r.converged and r.steps == 3 for r in results)
    assert [s for s, _ in seen] == [0, 1, 2, 3] and seen[0][1] == list(range(len(inputs)))
    for a, p in zip(inputs, before):
        assert np.array_equal(a.positions, p) and a.calc is None


if __name__ == "__main__":
    if pytest is not None:
        raise SystemExit(pytest.main([__file__, "-v"]))
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("passed", name)
