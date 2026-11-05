#!/usr/bin/env python
"""Step 0 (no DEM needed): build, validate and export the six executed graphs.

    python scripts/run_graphs.py

Writes outputs/graphs/<site>/{nodes.csv,edges.csv,descriptors.json,graph.png} and
outputs/graphs/validation.csv (every Table 4 check, pass/fail)."""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rtmap import graphs  # noqa: E402

specs = graphs.load_specs(ROOT / "config/graphs.yaml")
out = ROOT / "outputs" / "graphs"
rows = []
for key, spec in specs.items():
    G = graphs.build(spec)
    ok, msgs = graphs.validate(G, spec)
    d = graphs.descriptors(G)
    d["observations"] = {o["name"]: {"node": o["node"], "sources_reaching": graphs.passes_observation(G, o["node"])}
                         for o in spec["observations"]}
    d["source_groups_are_delineated_catchments"] = False
    d["table4_check_passed"] = ok
    d["table4_check_messages"] = msgs
    k = out / key
    k.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(graphs.node_table(G, spec)).to_csv(k / "nodes.csv", index=False)
    pd.DataFrame(graphs.edge_table(G)).to_csv(k / "edges.csv", index=False)
    pd.DataFrame(spec.get("evidence", [])).to_csv(k / "evidence_register.csv", index=False)
    json.dump(d, open(k / "descriptors.json", "w"), indent=2)
    graphs.draw(G, spec, k / "graph.png")
    rows.append({"site": key, "nodes": d["n_nodes"], "edges": d["n_edges"], "sources": d["n_sources"],
                 "storage": d["n_storage_functions"], "logical_distances": list(d["logical_distances"].values()),
                 "matches_table4": ok, "messages": "; ".join(msgs)})
    print(f"{key:10s} {'OK ' if ok else 'FAIL'} nodes={d['n_nodes']} edges={d['n_edges']} sources={d['n_sources']} "
          f"distances={list(d['logical_distances'].values())} {msgs}")
pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
sys.exit(0 if all(r["matches_table4"] for r in rows) else 1)
