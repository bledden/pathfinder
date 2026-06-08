"""Cycle-time-budget table: which decoders fit which hardware at d=6, R=6 on H200.

For each platform's per-block real-time budget, decide whether each decoder's
MEASURED latency (from the T9 manifest) fits. The binding question per platform:

  * SUPERCONDUCTING  ~1 us/round -> block budget = R us = 6 us (R=6). This is the
    HARD one and is CONCEDED TO FPGA in the paper (Relay-BP FPGA 24 ns/iter). A
    GPU decoder fits ONLY if it streams a block of R rounds within ~6 us. We
    report whether per-block latency (per-syndrome x R, treating a block as R
    syndromes' worth of work) fits, and at which batch the GPU kernel's amortized
    per-syndrome lands under the budget.
  * TRAPPED-ION      ~1 ms cycle  -> a decoder fits if its per-syndrome latency
    (or per-block, stated) is under ~1 ms.
  * NEUTRAL-ATOM     ~4.45 ms cycle (measured) -> the most forgiving; most
    ms-class decoders fit comfortably (often transversal, so per-block).

We are EXPLICIT about per-syndrome vs per-block and the batch dependence:

  - GPU decoders: per-syndrome latency is amortized over the batch (throughput),
    so the verdict is batch-dependent; we report the smallest batch at which the
    amortized per-syndrome fits each budget (and note that single-shot latency is
    higher -- the kernel needs a batch to hit its throughput).
  - CPU decoders: per-syndrome is roughly batch-independent (a python loop over
    shots), so the verdict is a flat comparison.
  - A "block" = R syndrome rounds; for a memory experiment the decoder must keep
    up with the syndrome-extraction RATE (throughput >= 1 block / cycle), which
    is the throughput/backlog framing (decode-rate >= syndrome-rate), the binding
    metric for memory (spec 9). We report both per-syndrome latency fit and the
    throughput/backlog fit.

Reads ``latency_results.json``; writes ``cycle_time_budget.json`` + prints.
"""
from __future__ import annotations

import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

R = 6  # rounds (= d) at the canonical grid point.

PLATFORMS = {
    "superconducting": dict(
        cycle_us=1.0,                 # per syndrome round
        block_budget_us=1.0 * R,      # block = R rounds
        note="conceded to FPGA (Relay-BP FPGA 24 ns/iter); GPU band shown for ref",
    ),
    "trapped-ion": dict(
        cycle_us=1.0e3,
        block_budget_us=1.0e3,        # ~1 ms per cycle
        note="~1 ms gate/measurement cycle",
    ),
    "neutral-atom": dict(
        cycle_us=4.45e3,
        block_budget_us=4.45e3,       # 4.45 ms measured cycle
        note="4.45 ms measured cycle; often transversal (per-block)",
    ),
}


def _decoder_latency(lat, name):
    """Return (us_per_syndrome, batch, is_gpu, sweep) for a decoder, or None."""
    rec = lat["results"].get(name)
    if not rec or rec.get("skipped") or rec.get("error"):
        return None
    if "batch_sweep" in rec:  # GPU
        rep = rec.get("representative")
        if not rep:
            return None
        ok = [s for s in rec["batch_sweep"] if "error" not in s]
        return dict(us_per_syndrome=rep["us_per_syndrome"], batch=rep["batch"],
                    is_gpu=True, sweep=ok,
                    throughput=rep["throughput_shots_per_s"])
    return dict(us_per_syndrome=rec["us_per_syndrome"], batch=rec["batch"],
                is_gpu=False, sweep=None,
                throughput=rec.get("throughput_shots_per_s"))


def _min_batch_under(sweep, budget_us):
    """Smallest GPU batch whose amortized us/syndrome is under budget_us."""
    fits = [s for s in sweep if s["us_per_syndrome"] <= budget_us]
    if not fits:
        return None
    return min(fits, key=lambda s: s["batch"])["batch"]


def build_table(lat):
    decoders = ["bp_triton", "bp_gpu", "BP", "BPOSD-0", "BPOSD-10", "BPLSD",
                "RelayBP", "SlidingWindow", "Tesseract"]
    rows = {}
    for name in decoders:
        info = _decoder_latency(lat, name)
        if info is None:
            rows[name] = dict(available=False)
            continue
        us_syn = info["us_per_syndrome"]
        us_block = us_syn * R       # per-block = R syndromes' work
        verdicts = {}
        for plat, spec in PLATFORMS.items():
            # per-SYNDROME budget = the per-round cycle (syndrome-rate match)
            syn_budget = spec["cycle_us"]
            block_budget = spec["block_budget_us"]
            fit_syn = us_syn <= syn_budget
            fit_block = us_block <= block_budget
            entry = dict(
                fits_per_syndrome=bool(fit_syn),
                fits_per_block=bool(fit_block),
                syndrome_budget_us=syn_budget,
                block_budget_us=block_budget,
            )
            if info["is_gpu"]:
                entry["min_batch_under_syndrome_budget"] = _min_batch_under(
                    info["sweep"], syn_budget)
                entry["min_batch_under_block_budget"] = _min_batch_under(
                    [dict(batch=s["batch"],
                          us_per_syndrome=s["us_per_syndrome"] * R)
                     for s in info["sweep"]], block_budget)
            verdicts[plat] = entry
        rows[name] = dict(
            available=True,
            is_gpu=info["is_gpu"],
            us_per_syndrome=us_syn,
            us_per_block=us_block,
            batch=info["batch"],
            throughput_shots_per_s=info["throughput"],
            verdicts=verdicts,
        )
    return rows


def summarize(rows):
    """Per-platform: which decoders fit (per-block), for the report."""
    summary = {}
    for plat in PLATFORMS:
        fit, nofit = [], []
        for name, r in rows.items():
            if not r.get("available"):
                continue
            v = r["verdicts"][plat]
            (fit if v["fits_per_block"] else nofit).append(name)
        summary[plat] = dict(fits_per_block=fit, does_not_fit=nofit)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="cycle-time budget table")
    ap.add_argument("--latency", default=os.path.join(_HERE, "latency_results.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "cycle_time_budget.json"))
    args = ap.parse_args(argv)

    with open(args.latency) as f:
        lat = json.load(f)

    rows = build_table(lat)
    summary = summarize(rows)
    out = dict(
        kind="cycle-time-budget",
        code=lat.get("code"), rounds=R, p=lat.get("p"), basis=lat.get("basis"),
        gpu=lat.get("env", {}).get("gpu_name"),
        platforms=PLATFORMS,
        per_decoder=rows,
        summary_fits_per_block=summary,
        framing=("'block' = R syndrome rounds; per-syndrome latency for GPU "
                 "decoders is amortized over the batch (batch-dependent verdict); "
                 "CPU decoders are ~batch-independent. The binding metric for a "
                 "memory experiment is throughput/backlog (decode-rate >= "
                 "syndrome-rate); per-syndrome latency fit + min-batch reported."),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- printed table ---- #
    print(f"\nCYCLE-TIME BUDGET  --  {out['code']} d={R} R={R} p={out['p']} "
          f"basis={out['basis']}  ({out['gpu']})")
    print("budgets (block = R rounds): "
          + ", ".join(f"{k} {v['block_budget_us']:g} us" for k, v in PLATFORMS.items()))
    print("-" * 100)
    print(f"{'decoder':<16}{'us/syn':>11}{'us/block':>11}"
          f"{'superconduct':>15}{'trapped-ion':>14}{'neutral-atom':>14}")
    print("-" * 100)
    for name, r in rows.items():
        if not r.get("available"):
            print(f"{name:<16}  -- unavailable")
            continue
        def cell(plat):
            v = r["verdicts"][plat]
            mark = "FIT" if v["fits_per_block"] else "no"
            if r["is_gpu"] and v.get("min_batch_under_block_budget"):
                mark += f"@b>={v['min_batch_under_block_budget']}"
            elif r["is_gpu"] and v["fits_per_block"]:
                mark += "@b"
            return mark
        print(f"{name:<16}{r['us_per_syndrome']:>11.3f}{r['us_per_block']:>11.2f}"
              f"{cell('superconducting'):>15}{cell('trapped-ion'):>14}"
              f"{cell('neutral-atom'):>14}")
    print("-" * 100)
    for plat, s in summary.items():
        print(f"{plat:<16} FITS (per-block): {', '.join(s['fits_per_block']) or 'NONE'}")
    print(f"\nwrote {args.out}")
    return out


if __name__ == "__main__":
    main()
