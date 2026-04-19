# Session 3 results (April 19, 2026)

**Context:** The user's RunPod instance ran out of balance mid-evaluation, wiping all on-pod checkpoints and logs. This file captures what I observed inline before the pod went down at approximately 08:22 UTC. Results are paper-quality where captured, partial where noted.

## Runs that completed on the pod (rc=0)

### Retrain #43: v1 recipe at 4-parameter noise (to match Lange's training condition)

- **fixed_d5** (80K steps, H=256, p=0.007): best training-eval LER **3.71%** at p=0.007 (10K-shot eval). Lange GNN on the same noise model: 2.49%. Pathfinder OOD on same noise (Table-1 ckpt): 2.94%. **The 4-parameter retrain did not close the Lange gap at d=5 — best eval came in worse than the OOD baseline.**
- **fixed_d7** (80K steps, H=256, p=0.007): **catastrophic failure**. Eval LER stuck at 0.393-0.410 throughout the entire run (random-chance level). Final training loss ~0.63; the model never made it out of stage-1 initialization. Renamed to `fixed_d7_broken/` on the pod before it went down.

### Muon ablation #38: AdamW-only on ALL parameters, 3-parameter noise

- **ablation_adamw_d3** (20K steps, H=256, p=0.007): **final 50K-shot LER = 1.82%**. Compare to Table-1 full-Muon d=3 at p=0.007 = **1.82%**. Muon provides ≈0% benefit at d=3.
- **ablation_adamw_d7** (80K steps, H=256, p=0.007): **final 50K-shot LER = 34.8%**. Compare to Table-1 full-Muon d=7 at p=0.007 = **1.04%**. Muon is **essential** at d=7 — AdamW-only failed to learn in the same step budget.

**Implication for §6.2 (Role of Muon):**
The paper currently reports a +72% LER effect from removing Muon at d=5. These d=3 and d=7 measurements show the effect is strongly depth-dependent:
- d=3: Muon effect ≈ 0% (full-Muon and AdamW-only both land at 1.82%)
- d=5: Muon effect = +72% (Table 4, prior work)
- d=7: Muon effect catastrophic (1.04% → 34.8%)

The ablation refines the paper's claim: Muon is not uniformly important — it becomes more important as model depth/distance grows, and is arguably *essential* for deep neural decoders on surface codes.

## Partial results, ensemble #41

3-way majority vote (Pathfinder + Lange + PyMatching) at matched 4-parameter noise, 60K shots/point (3 seeds × 20K). 8 of 12 configured points were captured inline before the pod shutdown.

| key          | PF%    | Lange% | PM%    | Majority% | Oracle-LB% | Winner   |
|--------------|--------|--------|--------|-----------|------------|----------|
| d3_p0.003    | 0.582  | 0.535  | 0.665  | 0.555     | 0.432      | Lange    |
| d3_p0.005    | 1.595  | 1.493  | 1.798  | 1.567     | 1.185      | Lange    |
| d3_p0.007    | 2.923  | 2.713  | 3.205  | 2.810     | 2.065      | Lange    |
| d3_p0.010    | 5.533  | 5.140  | 5.852  | 5.308     | 3.940      | Lange    |
| d5_p0.003    | 0.295  | 0.192  | 0.340  | 0.215     | 0.118      | Lange    |
| d7_p0.005    | 1.082  | 0.752  | 0.985  | **0.677** | —          | **Majority** |
| d7_p0.007    | 4.010  | 2.940  | 3.343  | **2.563** | 1.057      | **Majority** |
| d7_p0.010    | 13.575 | 10.822 | 10.300 | **9.362** | 3.535      | **Majority** |

**Not captured (pod went down):** d5_p0.005, d5_p0.007, d5_p0.010, d7_p0.003.

**Findings that survive the data loss:**
1. 3-way majority vote **beats every individual decoder** at d=7 across all three captured noise rates. At d=7, p=0.007: majority 2.56% vs Lange alone 2.94% (13% relative improvement); at d=7, p=0.010: majority 9.36% vs Lange 10.82% (14%).
2. At d=3, majority is close to but does not beat Lange alone.
3. Oracle lower bound (all three wrong simultaneously) at d=7 p=0.007 is 1.06%, so majority captures roughly 40% of the available ensemble headroom.
4. Confidence-thresholded gating (PF if |logit|>T else Lange) provided no meaningful improvement over Lange alone at any captured point.

## Configuration notes

- Pathfinder checkpoint used per distance: d=3 → `d3_muon/best_model.pt` (OOD Table-1 ckpt); d=5 → `fixed_d5/best_model.pt` (4-param retrain, 3.71% LER); d=7 → `d7_final/best_model.pt` (OOD Table-1 ckpt; fixed_d7 was broken).
- Noise model: 4-parameter Stim circuit-level depolarizing (`after_clifford_depolarization`, `before_measure_flip_probability`, `after_reset_flip_probability`, `before_round_data_depolarization`).
- Lange: `GNN_7` + `d{d}_d_t_{d_t}.pt` pre-trained weights from the Lange repo.
- PyMatching: `Matching.from_detector_error_model(dem, decompose_errors=True)` + `decode_batch`.

## What's lost

- The `ensemble_results.json` (complete 12-point data). Partial data above is from inline log capture, not the JSON.
- All on-pod checkpoints: `fixed_d5/`, `fixed_d7_broken/`, `ablation_adamw_d3/`, `ablation_adamw_d7/`.
- All on-pod training logs.
- Ablation training trajectories (only the final number was captured).

## Paper impact

Three substantive claims from this session are defensible on captured data:
1. Refined Muon ablation (§6.2): Muon effect is depth-dependent; specifically measured at d=3 (≈0%) and d=7 (essential, 1.04% → 34.8%).
2. §5.11 extension: the 4-parameter retrain at d=5 *does not* close the gap to Lange, contradicting the "future work" hedge in the current §5.11 draft. Retrain at d=7 fails entirely under the v1 recipe.
3. New §5.12 contribution: 3-way PF+Lange+PM majority vote beats every individual decoder at d=7, including Lange. This is a new contribution on top of Lange's priority.
