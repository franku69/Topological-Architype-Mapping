"""The six executed directed graphs: loading, validation against the paper's Table 4, descriptors.

A node is a named regional component or a declared aggregate; it is NOT a delineated catchment.
See config/graphs.yaml for the reading rules.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

ROLES = {"source", "storage", "junction", "zone", "probe", "receiver", "boundary"}


def load_specs(path):
    return yaml.safe_load(open(path))


def build(spec):
    """spec (one entry of graphs.yaml) -> networkx.DiGraph with node/edge attributes."""
    G = nx.DiGraph(name=spec["name"], receiver=spec["receiver"])
    for n, a in spec["nodes"].items():
        if a["role"] not in ROLES:
            raise ValueError(f"node {n}: unknown role {a['role']}")
        G.add_node(n, **a)
    for e in spec["edges"]:
        G.add_edge(e["from"], e["to"], delay=e.get("delay", True), frac=float(e.get("frac", 1.0)))
    return G


def sources(G):
    return [n for n, a in G.nodes(data=True) if a["role"] == "source"]


def logical_distances(G, receiver=None):
    """Source-to-receiver count of delay-bearing links (zero-delay connectors excluded).
    Splits that reconverge give the same count on both branches in these graphs; the longest is used."""
    receiver = receiver or G.graph["receiver"]
    out = {}
    for s in sources(G):
        best = None
        for path in nx.all_simple_paths(G, s, receiver):
            d = sum(1 for u, v in zip(path[:-1], path[1:]) if G.edges[u, v]["delay"])
            best = d if best is None else max(best, d)
        out[s] = best
    return out


def validate(G, spec):
    """Structural checks. Returns (ok, list of messages). Table 4 values come from the manuscript."""
    msgs, t4 = [], spec["table4"]
    if not nx.is_directed_acyclic_graph(G):
        msgs.append("graph is not acyclic")
    checks = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "sources": len(sources(G)),
              "storage": sum(1 for _, a in G.nodes(data=True) if a["role"] == "storage" and a.get("storage_multiplier", 1) > 0)}
    for k, v in checks.items():
        if k == "storage":          # Table 4 counts IMPLEMENTED special storage functions only
            if v != t4[k]:
                msgs.append(f"storage positions {v} != Table 4 {t4[k]}")
        elif v != t4[k]:
            msgs.append(f"{k} {v} != Table 4 {t4[k]}")
    ld = sorted(logical_distances(G).values(), reverse=True)
    if ld != sorted(t4["logical_distances"], reverse=True):
        msgs.append(f"logical distances {ld} != Table 4 {t4['logical_distances']}")
    for n in G.nodes:
        if G.nodes[n]["role"] != "boundary" and G.out_degree(n) == 0:
            msgs.append(f"node {n} has no outgoing edge")
        if G.nodes[n]["role"] == "source" and G.in_degree(n) > 0:
            msgs.append(f"source {n} has an incoming edge")
    for n in G.nodes:
        out = [d["frac"] for *_, d in G.out_edges(n, data=True)]
        if out and abs(sum(out) - 1) > 1e-12:
            msgs.append(f"outgoing fractions of {n} sum to {sum(out)}")
    reach = {s: nx.has_path(G, s, G.graph["receiver"]) for s in sources(G)}
    if not all(reach.values()):
        msgs.append(f"sources not reaching receiver: {[s for s, r in reach.items() if not r]}")
    return (not msgs), msgs


def passes_observation(G, obs_node):
    """Which sources can reach an observation node (Huong urban probe: Bo and residual cannot)."""
    return {s: nx.has_path(G, s, obs_node) for s in sources(G)}


def descriptors(G):
    """Topology metrics archived in the paper (section 3.3): counts, reachability, logical distances."""
    ld = logical_distances(G)
    vals = list(ld.values())
    return {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(), "n_sources": len(sources(G)),
        "n_junctions": sum(1 for _, a in G.nodes(data=True) if a["role"] in ("junction", "zone")),
        "n_terminal_boundaries": sum(1 for _, a in G.nodes(data=True) if a["role"] == "boundary"),
        "n_storage_functions": sum(1 for _, a in G.nodes(data=True) if a["role"] == "storage" and a.get("storage_multiplier", 1) > 0),
        "logical_distances": ld, "logical_distance_mean": sum(vals) / len(vals),
        "logical_distance_min": min(vals), "logical_distance_max": max(vals),
    }


def edge_table(G):
    return [{"from": u, "to": v, "delay_bearing": d["delay"], "fraction": d["frac"],
             "from_role": G.nodes[u]["role"], "to_role": G.nodes[v]["role"]} for u, v, d in G.edges(data=True)]


def node_table(G, spec):
    ev = {e["item"].split("->")[0]: e for e in spec.get("evidence", [])}
    rows = []
    for n, a in G.nodes(data=True):
        rows.append({"node": n, "role": a["role"], "label": a["label"], "aggregate": bool(a.get("aggregate", False)),
                     "provisional": bool(a.get("provisional", False)),
                     "is_delineated_catchment": False,   # by construction (paper 3.3, Table 4a)
                     "area_km2_published": (spec.get("area_km2") or {}).get(n)})
    return rows


def draw(G, spec, path):
    """Schematic (NOT a map): layered by longest path from sources, like Figures 5/16/29/43/58/72."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layer = {}
    for n in nx.topological_sort(G):
        layer[n] = 0 if G.in_degree(n) == 0 else 1 + max(layer[p] for p in G.predecessors(n))
    cols = {}
    for n, l in layer.items():
        cols.setdefault(l, []).append(n)
    pos = {n: (i - (len(ns) - 1) / 2, -l) for l, ns in cols.items() for i, n in enumerate(ns)}
    colour = {"source": "#ffffff", "storage": "#fdf0dc", "junction": "#f1f1f1", "zone": "#f1f1f1",
              "probe": "#e3eef8", "receiver": "#e3eef8", "boundary": "#ececec"}
    fig, ax = plt.subplots(figsize=(7.5, 1.2 + 1.1 * (max(layer.values()) + 1)))
    for u, v, d in G.edges(data=True):
        ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle="->", color="#444",
                    linestyle="-" if d["delay"] else ":", shrinkA=14, shrinkB=14))
    for n, (x, y) in pos.items():
        a = G.nodes[n]
        ax.text(x, y, f"{n}", ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", fc=colour[a["role"]], ec="#333",
                          ls="--" if a.get("aggregate") else "-"))
    ax.set_title(f"{spec['name']} - executed graph ({spec['figure']})\n"
                 "schematic, not a map | dashed box = aggregate | dotted arrow = zero-delay", fontsize=8)
    xs = [x for x, _ in pos.values()]; ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.03)
    fig.savefig(path, dpi=130)
    plt.close(fig)
