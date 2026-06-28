"""Section 5.7 phenomenological-noise generalization, ORIGINAL Table-1 Pathfinder.

Regenerates bench/results/h200_main/tierB/phenom_eval_table1.json: the original
3-parameter circuit-level-trained Pathfinder (the Table-1 checkpoints, per
run_final_eval.py) evaluated on phenomenological noise. Tests whether the
original Section 5.7 generalization claim holds under a larger-sample eval.
See _phenom_eval.py for the shared core.

Run:  python bench/results/h200_main/tierB/eval_phenom_table1.py
"""
import os
from _phenom_eval import cli

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# Table-1 original-training checkpoints (mapping from run_final_eval.py)
CKPTS = {3: f"{REPO}/train/checkpoints/best_model.pt",
         5: f"{REPO}/train/checkpoints/d5_muon/best_model.pt",
         7: f"{REPO}/train/checkpoints/d7_final/best_model.pt"}
NOTE = ("Table-1 Pathfinder (original 3-parameter circuit-level noise training) evaluated on "
        "phenomenological noise. Tests whether the ORIGINAL paper's Section 5.7 generalization "
        "claim holds. 60K shots per point.")

if __name__ == "__main__":
    cli(CKPTS, f"{REPO}/bench/results/h200_main/tierB/phenom_eval_table1.json", NOTE)
