# IBM Heron r2 real-hardware decoding

This directory holds the decoding pipeline for the real-hardware (IBM Heron r2 `ibm_fez` /
`ibm_kingston`) experiments reported in `paper/pathfinder.md` §5.15.1.

Published here:
- `decode_ibm_result.py` — decode raw IBM measurement records with PyMatching and Pathfinder.
- `redecode_ibm_d3r3_pfwl3s.py` — 3-seed PFWL3S logit-average ensemble on the d=3 r=3 data.
- `redecode_ibm_d5r5_pfwl3s.py` — 5-seed PFWL3S ensemble on the d=5 r=5 data.
- `eval_ibm_d7r7.py` — d=7 r=7 evaluation (the past-threshold `ibm_kingston` ceiling row).
- `ibm_full_eval.json` — per-shot decomposition (PM / PFWL3S / Lange / Triad) used in §5.15.1.

Raw measurement outcomes are in `bench/results/ibm_heron_r2/ibm_d{D}r{R}_result.json`.

Other exploratory device-characterization scripts (maximum-likelihood ceiling checks, soft-readout
and correlated-noise probes) were run locally and are not part of the paper's reported results; they
are kept out of the repository to avoid bloat.
