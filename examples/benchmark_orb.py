#!/usr/bin/env python
"""Speed and agreement: gpu_lbfgs.relax vs ase.optimize.LBFGS (one structure at a time), both with ORB.

    python examples/benchmark_orb.py structures.extxyz --n 512 --ase-n 64

The first --ase-n structures are also relaxed with ASE (LBFGS on FrechetCellFilter, ORBCalculator)
to compare steps and final energies; ORB on the GPU is not bit-reproducible, so expect small
differences rather than identical trajectories.
"""
import argparse
import time

import numpy as np
import torch
from ase.io import read

from gpu_lbfgs import orb_predict, relax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structures", help="file with many frames, readable by ase.io.read")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ase-n", type=int, default=64)
    ap.add_argument("--model", default="orb_v3_conservative_inf_mpa")
    ap.add_argument("--weights")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=10.0, help="L-BFGS alpha for both methods (ASE default: 70)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 512])
    args = ap.parse_args()

    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    kwargs = {"device": "cuda", "precision": "float32-high"}
    if args.weights:
        kwargs["weights_path"] = args.weights
    model, adapter = getattr(pretrained, args.model)(**kwargs)
    calc = ORBCalculator(model, adapter, device="cuda")
    predict = orb_predict(calc)
    atoms_list = read(args.structures, index=f":{args.n}")
    print(f"{len(atoms_list)} structures, {sum(map(len, atoms_list))} atoms; {torch.cuda.get_device_name()}; alpha {args.alpha}")

    relax(atoms_list[:8], predict, fmax=args.fmax, steps=5)           # warm-up
    timings = {}
    for bs in args.batch_sizes:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = relax(atoms_list, predict, fmax=args.fmax, steps=args.steps, batch_size=bs, alpha=args.alpha)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        steps = np.array([r.steps for r in results])
        timings[bs] = (dt, results)
        print(f"gpu_lbfgs batch {bs:4d}: {dt:7.1f} s = {len(results) / dt:6.2f} structures/s, "
              f"{steps.sum() / dt:7.1f} steps/s; converged {np.mean([r.converged for r in results]):.1%}, "
              f"mean steps {steps.mean():.1f}")

    best = min(timings, key=lambda b: timings[b][0])
    ours = timings[best][1][:args.ase_n]
    t0 = time.perf_counter()
    ref_e, ref_steps = [], []
    for atoms in atoms_list[:args.ase_n]:
        a = atoms.copy()
        a.calc = calc
        opt = LBFGS(FrechetCellFilter(a), alpha=args.alpha, logfile=None)
        opt.run(fmax=args.fmax, steps=args.steps)
        ref_e.append(a.get_potential_energy() / len(a))
        ref_steps.append(opt.nsteps)
    dt_ase = time.perf_counter() - t0
    print(f"ASE LBFGS one by one: {dt_ase:7.1f} s for {args.ase_n} = {args.ase_n / dt_ase:6.2f} structures/s, "
          f"mean steps {np.mean(ref_steps):.1f}")
    de = np.array([r.energy / len(r.atoms) - e for r, e in zip(ours, ref_e)]) * 1000
    print(f"speed-up (best batch {best}): {(len(timings[best][1]) / timings[best][0]) / (args.ase_n / dt_ase):.1f}x")
    print(f"final energy, gpu_lbfgs - ASE (meV/atom): median |dE| {np.median(np.abs(de)):.3f}, "
          f"max |dE| {np.abs(de).max():.3f}; same step count for {np.mean(np.array([r.steps for r in ours]) == ref_steps):.0%}")


if __name__ == "__main__":
    main()
