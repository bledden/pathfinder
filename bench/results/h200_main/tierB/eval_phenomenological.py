"""Section 5.7 phenomenological-noise generalization, CANONICAL fine-tune Pathfinder.

Regenerates bench/results/h200_main/tierB/phenom_eval.json: the canonical
4-parameter-fine-tuned Pathfinder (finetune_d{3,5,7}) evaluated on phenomenological
noise (data-qubit depolarizing only). Shows Pathfinder does NOT beat PyMatching on
this out-of-distribution noise model. See _phenom_eval.py for the shared core.

Run:  python bench/results/h200_main/tierB/eval_phenomenological.py
"""
import os
from _phenom_eval import cli

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CKPTS = {d: f"{REPO}/bench/results/h200_main/tuned/finetune_d{d}/best_model.pt"
         for d in (3, 5, 7)}
NOTE = ("Canonical Pathfinder (fine-tune at 4-parameter noise) evaluated on phenomenological "
        "noise (data-qubit errors only, no measurement errors). before_round_data_depolarization "
        "only. 60K shots per point (3 seeds x 20K).")

if __name__ == "__main__":
    cli(CKPTS, f"{REPO}/bench/results/h200_main/tierB/phenom_eval.json", NOTE)
