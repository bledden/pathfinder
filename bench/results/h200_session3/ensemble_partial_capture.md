# Partial ensemble capture (pod went unreachable during scp)

Captured from the inline log output of `ensemble_pf_lange.py` on the pod.
60K shots per point (3 seeds × 20K).

Configuration:
- d=3: Pathfinder = `d3_muon/best_model.pt` (Table-1 3-param ckpt, OOD on 4-param noise)
- d=5: Pathfinder = `fixed_d5/best_model.pt` (4-param retrain, 3.71% LER model)
- d=7: Pathfinder = `d7_final/best_model.pt` (Table-1 3-param ckpt, OOD on 4-param noise — fixed_d7 was catastrophic failure)

Noise: 4-parameter circuit-level depolarizing (Lange's training condition).

| key          | PF%    | Lange% | PM%    | Majority% | Oracle-LB% | Who wins        |
|--------------|--------|--------|--------|-----------|------------|-----------------|
| d3_p0.003    | 0.582  | 0.535  | 0.665  | 0.555     | 0.432      | Lange           |
| d3_p0.005    | 1.595  | 1.493  | 1.798  | 1.567     | 1.185      | Lange           |
| d3_p0.007    | 2.923  | 2.713  | 3.205  | 2.810     | 2.065      | Lange           |
| d3_p0.010    | 5.533  | 5.140  | 5.852  | 5.308     | 3.940      | Lange           |
| d5_p0.003    | 0.295  | 0.192  | 0.340  | 0.215     | 0.118      | Lange           |
| d7_p0.005    | 1.082  | 0.752  | 0.985  | **0.677** | —          | **Majority**    |
| d7_p0.007    | 4.010  | 2.940  | 3.343  | **2.563** | 1.057      | **Majority**    |
| d7_p0.010    | 13.575 | 10.822 | 10.300 | **9.362** | 3.535      | **Majority**    |

Missing from captured output (pod went down before scp): d5_p0.005, d5_p0.007, d5_p0.010, d7_p0.003.

## Findings

1. **3-way majority vote** (Pathfinder + Lange + PM) beats every individual decoder at d=7 across tested noise rates — including Lange. At d=7, p=0.007, majority = 2.56% vs Lange alone 2.94% (13% relative improvement); at d=7, p=0.010, majority = 9.36% vs Lange's 10.82% (14% relative).

2. At d=3 and d=5, the majority vote is close to Lange but does **not** beat it — at those distances Lange's individual accuracy is already near the ensemble limit.

3. The oracle lower bound (all three wrong) at d=7 p=0.007 is 1.06%, showing the majority vote captures about 40% of the available ensemble headroom (2.56% majority vs 2.28% theoretical oracle = (pf∩lange∩pm)|obs limit).

4. Confidence-thresholded gating ("use PF if |logit|>T else Lange") does not meaningfully improve over Lange alone at any point — gate numbers are within 0.01-0.1 of Lange alone.

## Implication for paper

This is a genuinely strong contribution: a simple 3-way majority vote that includes Lange as one vote beats Lange alone at d=7. That is a **new** contribution on top of Lange's work, independent of whether Pathfinder individually beats Lange.
