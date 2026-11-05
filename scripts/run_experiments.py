#!/usr/bin/env python
"""Step 2: the paper's abstract routing experiment on the six graphs (sections 3.11-3.19).
Needs no DEM and no internet.     python scripts/run_experiments.py

Prints this reconstruction next to the values REPORTED in the manuscript (Tables 136/137), so any
difference caused by unrecorded original details is visible rather than hidden."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rtmap import graphs, routing  # noqa: E402

REPORTED = {  # manuscript Table 136 (mean spill %, any-spill %) and Table 137 (paired timing, pp)
    "cdo": (18.39, 49.88, 7.49), "manado": (22.05, 55.04, 7.96), "kuranji": (20.33, 52.21, 12.82),
    "huong": (17.18, 47.21, 6.76), "kadamaian": (22.00, 54.90, 7.66), "hat_yai": (17.95, 48.47, 13.21)}
specs = graphs.load_specs(ROOT / "config/graphs.yaml")
rows = []
for key, spec in specs.items():
    G = graphs.build(spec)
    ref = routing.run_reference(G)
    s = routing.summarize(ref)
    t = routing.paired_timing(G)
    rep = REPORTED[key]
    rows.append({"site": key, "receiver_evals": s["n_receiver_evaluations"],
                 "mean_spill_pct": round(s["mean_spill_pct"], 2), "paper_mean_spill_pct": rep[0],
                 "any_spill_pct": round(s["any_spill_share_pct"], 2), "paper_any_spill_pct": rep[1],
                 "timing_pp": round(t["mean_change_pp"], 2), "paper_timing_pp": rep[2],
                 "timing_pairs": t["n_pairs"], "timing_lower_pct": round(t["lower_pct"], 2),
                 "synchrony_chi_median": round(float(pd.Series(ref["chi"]).median()), 3),
                 "max_conservation_error": routing.conservation_error(ref)})
df = pd.DataFrame(rows)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
df.to_csv(out / "reconstruction_vs_reported.csv", index=False)
print(df.to_string(index=False))
