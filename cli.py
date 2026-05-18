#!/usr/bin/env python3
"""Pathfinder command-line dispatcher.

Unified CLI entry point routing to the various evaluation and training
scripts at the repo root. The existing top-level ``run_*.py`` scripts
remain in place for backwards compatibility — this dispatcher is the
recommended entry point for new users and the README's Quick Start.

USAGE:
    python cli.py <subcommand> [args...]
    python cli.py --help              # list all subcommands
    python cli.py eval-table1         # reproduce paper Table 1 (Section 5.1)
    python cli.py eval-comprehensive  # broader sweep with error bars (§5.1–5.2)
    python cli.py eval-100k           # mixed-noise + ensemble eval (§4.5)
    python cli.py eval-code-types     # code-type generalization eval (§5.7)
    python cli.py train-mixed         # mixed-noise training (§4.5)
    python cli.py train-p015          # d=7 specialized at p=0.015 (§4.5)

Every subcommand is a thin wrapper around an existing run_*.py script;
the dispatcher exists to give the audit reviewer a single ``--help``
landing page that documents what produces which paper section.
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUBCOMMANDS = {
    "eval-table1": {
        "script": "run_final_eval.py",
        "produces": "paper §5.1 Table 1 (Pathfinder vs PyMatching at 3-parameter noise, 100K shots, d=3/5/7 × 8 noise rates)",
    },
    "eval-comprehensive": {
        "script": "run_comprehensive_eval.py",
        "produces": "broader Table-1-style sweep with Wilson-CI error bars, scaling ratios (§5.1, §5.2)",
    },
    "eval-100k": {
        "script": "run_100k_eval.py",
        "produces": "mixed-noise model + per-noise-target ensemble eval at 100K shots (§4.5)",
    },
    "eval-code-types": {
        "script": "run_code_types.py",
        "produces": "color-code, rotated surface code X/Z basis generalization eval (§5.7, Table 6)",
    },
    "train-mixed": {
        "script": "run_mixed_noise.py",
        "produces": "mixed-noise training (uniform sampling of p across an interval; §4.5)",
    },
    "train-p015": {
        "script": "run_d7_p015.py",
        "produces": "d=7 model specialized at p=0.015 (the noise-rate-specialization recipe of §4.5)",
    },
}


def list_subcommands():
    print("\nAvailable subcommands:\n")
    width = max(len(k) for k in SUBCOMMANDS)
    for name, meta in SUBCOMMANDS.items():
        print(f"  {name:<{width}}  → {meta['script']}")
        print(f"  {'':<{width}}    produces: {meta['produces']}")
        print()
    print("Run `python cli.py <subcommand> [--help]` to invoke. Args after the")
    print("subcommand are forwarded to the underlying script verbatim.\n")


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.split("USAGE:")[0].strip())
        list_subcommands()
        return 0
    sub = argv[0]
    if sub not in SUBCOMMANDS:
        print(f"Unknown subcommand: {sub!r}\n", file=sys.stderr)
        list_subcommands()
        return 2
    script_path = ROOT / SUBCOMMANDS[sub]["script"]
    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 1
    # Forward remaining argv to the script as if it were invoked directly.
    sys.argv = [str(script_path)] + argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
