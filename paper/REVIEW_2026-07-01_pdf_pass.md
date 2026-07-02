# Multi-agent PDF review — pathfinder_draft.pdf (2026-07-01)

**Artifact:** `paper/pathfinder_draft.pdf`, 56 pp, built 2026-06-29 18:39 at `f5d2cc9` (md5 `776ba352…`, identical to `~/Documents/pathfinder_draft.pdf`).
**Method:** 18 agents — 4 overlapping page-range readers + 1 global claims reader (each with the R1–R9 revision-residue checklist + recompute-every-stat-from-printed-counts rule + formatting lens), a pooled-number-registry consistency pass (125 numbers), then adversarial verification (verifier re-reads the cited pages and tries to REFUTE; default-refute when uncertain). 80 raw findings → 41 blocker/major candidates → 12 workflow-verified (10 confirmed, 2 refuted) + 4 verified inline by the session (pp. 3, 31, 44, 48). ~703K subagent tokens.

## The systemic pattern (read this first)

**The figures are stale relative to the revised text and data.** Both blockers and several majors are numbers/verdicts baked into figure images (titles, subtitles, annotations, grid cells) that the text-level claim-discipline revision could not touch. Every defect found in a figure contradicts body text that is CORRECT. Ironically, both blockers UNDERSELL the results (they show losses/ties where the data shows wins). Fix = regenerate figures from current data (`figures/make_figures.py`; note the hardcoded title at line ~596), not text edits.

## BLOCKERS (2 — both stale figures, both verified)

**B1. Figure 2 (p.11) asserts the headline claim is false; it isn't.** Title/subtitle: "10 strict-CI wins / 13 statistical ties / 1 strict-CI loss … only strict-CI loss is at d=7, p=0.015." Recomputing Wilson CIs from Table 1 counts (100K shots): **12 strict wins / 12 ties / 0 losses** — matching §5.1's text exactly. The claimed "loss" cell (d=7 p=0.015: PF 15.546% vs PM 16.883%) is a strict PF **WIN** (CI edges 15.77 vs 16.65). Panel tallies also wrong (d=5 should be 6W/2T/0L not 5W/3T/0L; d=7 4W/4T/0L not 3W/4T/1L). The figure even contradicts its own caption. The phantom loss appears imported from Figure 1's PFWL3S note ("PM beats PFWL3S by 0.18 pp").

**B2. Figure 6 (p.26) 'PFWL3S vs Lange-pub' column is built from mixed/stale data; d=5 win/loss direction inverted.** Shows strict losses at d=5 (−0.22 at p=0.007, −0.90 at p=0.015) where Table 11 (p.28) + the p.30 CIs say **tie at p=0.007, strict WIN at p=0.015** (17.205% [16.972,17.440] vs 17.898% [17.662,18.137] — 0.222 pp non-overlap). Mechanism confirmed: the column's d=7 cells match genuine 3-seed PFWL3S gaps (0.262/1.219/2.311), the d=5 cells match single-seed canonical Pathfinder (~0.90 pp), and the column has d=9 cells although PFWL3S was never evaluated at d=9. The caption says these columns "define this paper's headline claims."

## MAJORS (verified)

**M1 (p.2, abstract).** "0.262 pp CI-edge gap" for PFWL3S vs **Lange-FT** is ~5.3× overstated (true ≈0.049 pp). 0.262 recomputes exactly as the gap vs **published** Lange (2.956%) from earlier in the same abstract — copy-paste. Non-overlap claim survives.
**M2 (p.15, §5.4).** "72% LER reduction compared to AdamW" — direction/arithmetic error. Table 4 (same page): removing Muon **increases** LER 72% (1.28→2.20); Muon's reduction is **42%**. §4.2 (p.8) states it correctly.
**M3 (p.9, Figure 1 caption).** "strict-CI dominance over … PFWL3S alone" is false at the headline point (Triad 2.384% [2.291,2.480] vs PFWL3S 2.492% [2.397,2.590] — OVERLAP), and the vs-Lange strict-win set doesn't include p=0.005. Abstract scopes it correctly; caption drops the scoping.
**M4 (p.4, Contribution 2).** PFWL3S "~24 µs/syn extrapolated" is the pre-audit figure the paper itself retracts in §5.13 (audit-measured ≈61 reference / ≈77 Triton). Understates by ~2.5×.
**M5 (p.26, Figure 6 title).** "Triad strict-CI beats Lange at 5 of 8 d≥7 ops points" — the rendered grid shows **7 of 8** green WIN cells (6 discounting the +0.00 cell). "5 of 8" = the pre-d=9 grid count; hardcoded at `make_figures.py:596`.
**M6 (p.25, Table 10 vs Table 9 p.20).** d=7 PF column disagrees at p=0.003 (0.120 vs 0.103) and p=0.005 (0.965 vs 0.817) while PM/Lange columns are IDENTICAL (same syndromes → a deterministic decoder must match). Verified against raw: Table 10 = `tuned/ensemble_results_tuned.json`, Table 9 = a **splice** (phase2 at low rates, tuned at high). Same-checkpoint claim contradicted; touches the starred d=7 p=0.005 Triad cell. Needs provenance reconciliation.
**M7 (p.19, §5.10).** "17% relative improvement" not derivable (2.855→2.520 = 11.7%; 13.3% normalized). Implies a stale ~3.04% baseline.
**M8 (p.44 + Table 14, d=9; session-verified).** Both rates mislabel CI-edge separations as "pp gap" mixed with point-estimate percentages: p=0.007 "0.154 pp gap, 13.2% relative" (point gap 0.346; 0.154 = CI-edge), p=0.010 "1.831 pp gap, 17.1%" (point gap 2.233; 1.831 = exactly Lange-lower − Triad-upper). Strict wins themselves are real. p.48 shows the correct labeling ("0.372 pp non-overlapping-Wilson-CI separation") — adopt it.
**M9 (p.31, §5.13; session-verified).** "For **real-time** d=7 decoding **inside the 7-µs cycle budget** … (6.12 µs/syn) … in exchange for **cycle-time compliance**" — built on B=1024 *amortized throughput*; same page shows batch-1 ≈ ms. The honest "throughput-sustainability, not streaming-latency" qualifier (p.14) is absent here. R8 residue.
**M10 (p.48, Conclusion; session-verified).** "wins or ties PyMatching at all 24/24 points (20 point-estimate wins … 3 ties …)" — 20+3=23; the 24th (d=7 p=0.002, PF 5 vs PM 4 fails) is a point-estimate loss the paper concedes on p.9; abstract says "20 of 24". Safe form: "never statistically beaten at any of the 24 points."
**M11 (p.2/p.3 abstract; session-verified, softer).** PFWL3S at d=7 p=0.007 appears as 2.492% (marginal 100K) and 2.533% (paired C3 run) in one abstract with no inline reconciliation. One parenthetical fixes it.

## HIGH-CONFIDENCE, NOT ADVERSARIALLY VERIFIED (multi-reader corroborated)

- **Fig 10 (p.43) in-image title**: "Muon is **essential** at d=7" — banned word baked into the image (caption is compliant). Regenerate.
- **Fig 11 (p.45) annotation**: "Triad 36.7%" vs Table 14's 35.701% [35.404,35.998] — likely 35.7 typo; regenerate.
- **§6.2 (p.42)**: "large at d≥7 … small at d=3" should be **d≥5** (same paragraph calls the d=5 +72% "the headline Muon result"; p.48 has the correct form).
- **R3 at claim sites (pp.43–45)**: the d=9 STRICT-WIN sites never carry the "published OOD Lange, no fairness control" flag (it appears only in the p.48 Conclusion; contrast the exemplary IBM caveat on p.36).
- **App B (p.55)**: three values for the same §5.12 Triad quantity (2.454 / 2.417 / 2.45); d=3 fine-tune cited as 2.87 / 2.817 / 2.77; "marginally improving …at p=0.007 2.448%" is a direction error vs 2.417%; "the per-noise-rate fine-tuning the §6.3 d=9 extension still needs" reintroduces the R7 framing and contradicts p.43's single-recipe description.
- **Fig 6 (p.26) "+0.00 pp WIN" cell** (d=7 p=0.005): a strict-CI win with 0.00 edge separation is self-contradictory; Table 10 counts give clear overlap. Inflates the 7-of-8 count.
- **p.18**: Table 7 header says "20K-shot" but every value is only expressible at 100K granularity (stale header); §5.7 "PM wins at 15/15 points … gap 1.5–8.5×" vs Table 6a's 7 printed rows and a printed 1.12× ratio.

## REFUTED BY VERIFICATION (do not "fix")

- **§4.6 cost table + "$550 over ~6 weeks" (p.5/p.8)** — NOT R9 residue: the memoir-trim commit (`57484a0`) deliberately scoped extraction to the §5.0 discovery-arc narrative + the timeline sentence and **kept** the cost table. Intentional.
- **Fig 6 Triad-vs-Lange d=9 p=0.005 cell** — computed correctly from its disclosed 100K dataset (reader had used the 60K table).

## MINORS (unverified; sweep during the fix pass)

"essentially ties PM" phrasing (pp.2, 4, 20, 45, 48 — R8 letter-violation, quantified in substance); p.5 cites the 0.01% PF–PM overlap bare (full 5/50K caveat exists on p.16 — add "5 shots" inline); p.14 "the only decoder … both real-time and accurate"; 250.8 vs 250.1 µs (p.15 vs Table 3a); Fig 2 caption "below or on … at every cell" false at d=7 p=0.002; Table 3b title orphaned (p.12/13); Fig 3 right-panel tick-label collision (eyeball print); three different checkpoints called "canonical" (pp.11/15/16); Fig 5 callout vs C3 text (27.16/27.33 vs 27.269/27.304, and 0.18 vs 0.17 arithmetic); Fig 5 caption "every d≥7 rate" should be "p≥0.007" (FT-Lange only exists at d=7); scattered "earlier draft" self-references (pp.43, 44, 56).

## Residue checklist verdict (R1–R9)

R1 **fixed** (pp.34–35 "~1.2× harder", correct direction; no 6× anywhere). R2 **fixed** (primary endpoint + Holm/Bonferroni correctly declared; honest p=0.015 tie). R3 **partial** (conclusion flagged; claim sites pp.43–45 not). R4 **partial** (p.8/p.48 correct; p.42 scopes to d≥7; p.15 has the inverted 72%). R5 **partial** (p.16 full caveat; p.5 bare). R6 **near-fixed** (p.16 calls d7_final "canonical"; p.10 disclosure wording borderline). R7 **near-fixed** (p.55 residue). R8 **partial** (ties-PM ×5, p.31 real-time, Fig 10 "essential"). R9 **fixed-by-design** (cost table deliberately kept; minor "earlier draft" self-refs remain).

## Title verdict

**Supported.** Triad has strict-CI wins vs published Lange at 5 (d,p) points with a disclosed fairness control; the Bonferroni-robust paired win is correctly attributed to the PFWL3S primary endpoint; limitations (off-budget latency, device-ceiling hardware, d=9 individual negative) stated honestly in abstract + conclusion.

## Recommended fix order

1. **Regenerate all stale figures** (Figs 2, 6, 10, 11 + Fig 1 caption + Fig 5 callout/caption) from current data — kills B1, B2, M3, M5 and three high-confidence items at once. Audit `make_figures.py` for hardcoded titles/annotations while in there.
2. One-line stats fixes: M1, M2, M7, M8 (use p.48's labeling), M10, M11.
3. Consistency/provenance: M6 (Table 9/10 splice — decide the canonical run and re-derive), M4, App B p.55 value reconciliation.
4. Residue sweep: R3 flags at claim sites, p.42 d≥5, p.31 qualifier, ties-PM phrasing, minors.
5. Rebuild PDF (`python paper/build_pdf.py`) and re-run this review's Fig-2/Fig-6 recomputation as the regression gate.
