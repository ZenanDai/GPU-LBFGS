#!/usr/bin/env python
"""Relax structure files with ORB on the GPU.

    python examples/relax_with_orb.py a.cif b.cif --out relaxed.extxyz
    python examples/relax_with_orb.py many.extxyz --fmax 0.02 --batch-size 256
"""
import argparse

from ase.io import read, write

from gpu_lbfgs import orb_predict, relax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structures", nargs="+", help="files readable by ase.io.read (all frames are used)")
    ap.add_argument("--model", default="orb_v3_conservative_inf_mpa", help="orb_models pretrained model")
    ap.add_argument("--weights", help="checkpoint file (default: the model's own download/cache)")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--positions-only", action="store_true", help="keep the cell fixed")
    ap.add_argument("--out", help="write the relaxed structures, e.g. relaxed.extxyz")
    args = ap.parse_args()

    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    kwargs = {"device": "cuda", "precision": "float32-high"}
    if args.weights:
        kwargs["weights_path"] = args.weights
    model, adapter = getattr(pretrained, args.model)(**kwargs)
    predict = orb_predict(ORBCalculator(model, adapter, device="cuda"))

    atoms_list = [a for path in args.structures for a in read(path, index=":")]
    results = relax(atoms_list, predict, fmax=args.fmax, steps=args.steps,
                    relax_cell=not args.positions_only, batch_size=args.batch_size)
    for i, r in enumerate(results):
        if r.atoms is None:
            print(f"{i}: failed ({r.error})")
        else:
            n = len(r.atoms)
            print(f"{i} {r.atoms.get_chemical_formula()}: {'converged' if r.converged else 'step limit'} "
                  f"after {r.steps} steps, {r.energy_initial / n:.4f} -> {r.energy / n:.4f} eV/atom")
    if args.out:
        write(args.out, [r.atoms for r in results if r.atoms is not None])


if __name__ == "__main__":
    main()
