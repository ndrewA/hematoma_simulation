"""Run all preprocessing steps in sequence.

Thin orchestrator that calls each step's main(argv) in order:
  1. Domain Geometry   (resample_sources)
  2. Label Remapping   (material_map)
  3. Skull SDF         (skull_sdf)
  4. Subarachnoid CSF  (subarachnoid_csf)
  5. Dural Membrane    (dural_membrane)
  6. Fiber Orientation  (fiber_orientation)
  7. Validation        (validation)

Usage:
    python -m preprocessing.run_all --subject 157336 --profile debug
"""

import argparse
import importlib
import sys
import time

from preprocessing.utils import PROFILES


STEPS = [
    ("1. Domain Geometry",          "preprocessing.resample_sources"),
    ("2. Label Remapping",          "preprocessing.material_map"),
    ("3. Skull SDF",                "preprocessing.skull_sdf"),
    ("4. Subarachnoid CSF",         "preprocessing.subarachnoid_csf"),
    ("5. Dural Membrane",           "preprocessing.dural_membrane"),
    ("6. Fiber Orientation",        "preprocessing.fiber_orientation"),
    ("7. Cross-cutting Validation", "preprocessing.validation"),
]


def parse_args(argv=None):
    """Parse CLI arguments for run_all."""
    parser = argparse.ArgumentParser(
        description="Run all preprocessing steps in sequence."
    )
    parser.add_argument("--subject", required=True, help="HCP subject ID")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        help="Named profile (default: debug)",
    )
    group.add_argument("--dx", type=float, help="Grid spacing in mm (custom)")

    parser.add_argument(
        "--grid-size", type=int,
        help="Grid size N (required with --dx, ignored with --profile)",
    )

    args = parser.parse_args(argv)

    if args.profile is None and args.dx is None:
        args.profile = "debug"

    if args.profile is not None:
        args.N, args.dx = PROFILES[args.profile]
    else:
        if args.grid_size is None:
            parser.error("--grid-size is required when using --dx")
        args.N = args.grid_size
        args.profile = f"custom_{args.N}_{args.dx}"

    return args


def build_step_argv(step_module, subject, profile):
    """Build argv list for a step's main().

    fiber_orientation is profile-independent (no --profile arg).
    """
    if step_module == "preprocessing.fiber_orientation":
        return ["--subject", subject]
    return ["--subject", subject, "--profile", profile]


def main(argv=None):
    """Run all preprocessing steps."""
    args = parse_args(argv)
    subject = args.subject
    profile = args.profile

    print("=" * 60)
    print(f"  Preprocessing Pipeline: {subject} / {profile}")
    print(f"  Grid: {args.N}^3, dx={args.dx} mm")
    print("=" * 60)

    t_total = time.monotonic()

    for step_name, module_name in STEPS:
        print(f"\n{'─' * 60}")
        print(f"  {step_name}")
        print(f"{'─' * 60}\n")

        t_step = time.monotonic()

        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            print(f"FATAL: Could not import {module_name}: {e}")
            sys.exit(1)

        step_argv = build_step_argv(module_name, subject, profile)

        try:
            mod.main(step_argv)
        except SystemExit as e:
            if e.code is not None and e.code != 0:
                elapsed = time.monotonic() - t_step
                print(f"\nFATAL: {step_name} exited with code {e.code} "
                      f"after {elapsed:.1f}s")
                sys.exit(e.code)
        except Exception as e:
            elapsed = time.monotonic() - t_step
            print(f"\nFATAL: {step_name} raised {type(e).__name__}: {e} "
                  f"after {elapsed:.1f}s")
            sys.exit(1)

        elapsed = time.monotonic() - t_step
        print(f"\n  [{step_name} completed in {elapsed:.1f}s]")

    total = time.monotonic() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete: {total:.1f}s total")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
