"""Stream network -> reach graph, GIS *catchment support* for the conceptual source groups, descriptors.

IMPORTANT (reviewer comment 4): the source groups U/T/B/L of the paper's graphs are CONCEPTUAL
aggregations (see graphs.py and config/graphs.yaml). The catchments delineated here are a separate,
optional GIS screening layer that shows how much terrain a given group could correspond to. They are not
inputs to the paper's routing experiment and are not claimed to be an authoritative delineation.
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from .hydro import upstream_mask


def stream_mask(acc, member, land, thr_cells):
    return (acc >= thr_cells) & member.ravel() & land.ravel()


def build_graph(recv, stream, acc, z, grid, shape, outlet, dd):
    """Directed reach graph. Nodes = sources, junctions, outlet (flat cell index)."""
    R, C = shape
    n = stream.size
    idx = np.flatnonzero(stream)
    j = recv[idx]
    ok = j >= 0
    ok[ok] = stream[j[ok]]
    inflow = np.zeros(n, int)
    np.add.at(inflow, j[ok], 1)
    is_node = stream & (inflow != 1)
    is_node[outlet] = True
    zf = z.ravel()
    cell_km2 = lambda i: grid.area[i // C] / 1e6
    G = nx.DiGraph()
    for i in np.flatnonzero(is_node):
        kind = "outlet" if i == outlet else ("source" if inflow[i] == 0 else "junction")
        lon, lat = grid.lonlat(i // C, i % C)
        G.add_node(int(i), kind=kind, lon=lon, lat=lat, area_km2=float(acc[i] * cell_km2(i)),
                   elev_m=float(zf[i]), dist_to_outlet_m=float(dd[i]))
    for i in np.flatnonzero(is_node):
        if i == outlet:
            continue
        path, cur = [int(i)], int(i)
        end = None
        while True:
            nxt = int(recv[cur])
            if nxt < 0 or not stream[nxt]:
                break
            path.append(nxt)
            cur = nxt
            if is_node[nxt]:
                end = nxt
                break
        if end is None:
            continue
        length = float(dd[i] - dd[end])
        drop = float(zf[i] - zf[end])
        a_cells = acc[path[-2]] if len(path) > 1 else acc[path[0]]
        G.add_edge(int(i), end, length_m=length, drop_m=drop,
                   slope=drop / length if length > 0 else 0.0,
                   area_km2=float(a_cells * cell_km2(path[-2] if len(path) > 1 else path[0])),
                   cells=path)
    _strahler(G)
    _upstream_sources(G)
    return G


def _strahler(G):
    order_node = {}
    for u in nx.topological_sort(G):
        ins = [G.edges[p, u]["strahler"] for p in G.predecessors(u)]
        if not ins:
            o = 1
        else:
            m = max(ins)
            o = m + 1 if ins.count(m) >= 2 else m
        order_node[u] = o
        for v in G.successors(u):
            G.edges[u, v]["strahler"] = o


def _upstream_sources(G):
    cnt = {}
    for u in nx.topological_sort(G):
        preds = list(G.predecessors(u))
        cnt[u] = 1 if not preds else sum(cnt[p] for p in preds)
        for v in G.successors(u):
            G.edges[u, v]["upstream_sources"] = cnt[u]
    return cnt


def descriptors(G, basin_area_km2):
    """Topology descriptors used in the paper (computed on the extracted graph)."""
    kinds = nx.get_node_attributes(G, "kind")
    src = [n for n, k in kinds.items() if k == "source"]
    jun = [n for n, k in kinds.items() if k == "junction"]
    L = sum(d["length_m"] for *_, d in G.edges(data=True)) / 1000.0
    d_src = np.array([G.nodes[s]["dist_to_outlet_m"] for s in src]) / 1000.0 if src else np.array([0.0])
    orders = [d["strahler"] for *_, d in G.edges(data=True)]
    maxo = max(orders) if orders else 0
    n_by_o = {o: orders.count(o) for o in range(1, maxo + 1)}
    rb = [n_by_o[o] / n_by_o[o + 1] for o in range(1, maxo) if n_by_o.get(o + 1)]
    asym = []
    for j in jun:
        a = sorted(G.edges[p, j]["area_km2"] for p in G.predecessors(j))
        if len(a) >= 2 and a[-1] > 0:
            asym.append(a[0] / a[-1])
    n_src = max(len(src), 1)
    bet = {(u, v): d["upstream_sources"] / n_src for u, v, d in G.edges(data=True)}
    return {
        "n_sources": len(src), "n_junctions": len(jun), "n_reaches": G.number_of_edges(),
        "max_strahler_order": maxo,
        "mean_bifurcation_ratio": float(np.mean(rb)) if rb else None,
        "stream_length_km": L,
        "drainage_density_km_per_km2": L / basin_area_km2 if basin_area_km2 else None,
        "confluence_density_per_100km2": 100 * len(jun) / basin_area_km2 if basin_area_km2 else None,
        "source_to_outlet_km_mean": float(d_src.mean()),
        "source_to_outlet_km_cv": float(d_src.std() / d_src.mean()) if d_src.mean() else None,
        "source_to_outlet_km_min": float(d_src.min()), "source_to_outlet_km_max": float(d_src.max()),
        "mean_junction_branch_asymmetry": float(np.mean(asym)) if asym else None,
        "max_edge_betweenness": max(bet.values()) if bet else None,
    }


# ----------------------------------------------------------------------- source groups
def _donors(recv, stream):
    idx = np.flatnonzero(stream)
    d = {}
    for i in idx.tolist():
        j = int(recv[i])
        if j >= 0 and stream[j]:
            d.setdefault(j, []).append(i)
    return d


def auto_source_points(recv, stream, acc, outlet, k):
    """Pick k source-group points: the k-1 largest tributaries that join the main stem,
    plus the main-stem point just upstream of the lowest selected confluence."""
    donors = _donors(recv, stream)
    cur, step, cands = outlet, 0, []
    while True:
        ds = sorted(donors.get(cur, []), key=lambda d: acc[d], reverse=True)
        if not ds:
            break
        for d in ds[1:]:
            cands.append((int(acc[d]), d, step, ds[0]))
        cur, step = ds[0], step + 1
    cands.sort(reverse=True)
    sel = cands[:max(k - 1, 0)]
    pts = [(f"tributary_{n+1}", s[1]) for n, s in enumerate(sel)]
    if sel:
        lowest = min(sel, key=lambda s: s[2])
        main_pt = lowest[3]
    else:
        main_pt = max(donors.get(outlet, [outlet]), key=lambda d: acc[d]) if donors.get(outlet) else outlet
    return [("main_stem", main_pt)] + pts


def delineate_groups(points, recv, order, member, shape):
    """points: [(name, flat_idx)]. Returns ({name: mask}, residual_mask).
    Nested catchments are subtracted so groups partition the basin; whatever is not in
    any group is the distributed lateral/residual area (reviewer comment 4)."""
    ups = {n: upstream_mask(recv, order, p, shape) & member for n, p in points}
    ascending = sorted(ups, key=lambda n: ups[n].sum())
    assigned = np.zeros(shape, bool)
    out = {}
    for n in ascending:
        out[n] = ups[n] & ~assigned
        assigned |= ups[n]
    return out, member & ~assigned


def group_table(points, masks, residual, recv, z, dd, grid, shape, outlet):
    """Subbasin parameters ready for HEC-HMS (screening values, see README)."""
    R, C = shape
    zf, ddf = z.ravel(), dd.ravel()
    rows = []
    for name, p in points:
        m = masks[name]
        cells = np.flatnonzero(m.ravel())
        head = cells[np.argmax(ddf[cells])]
        L = float(ddf[head] - ddf[p])
        drop = float(zf[head] - zf[p])
        S = drop / L if L > 0 else np.nan
        tc = 0.0195 * (L ** 0.77) * (S ** -0.385) if L > 0 and S > 0 else np.nan  # Kirpich, min
        lon, lat = grid.lonlat(p // C, p % C)
        j = int(recv[p])
        jl = grid.lonlat(j // C, j % C) if j >= 0 else (np.nan, np.nan)
        rows.append({"element": name, "type": "screening_catchment_support", "area_km2": grid.area_km2(m),
                     "point_lon": lon, "point_lat": lat, "confluence_lon": jl[0], "confluence_lat": jl[1],
                     "longest_flow_path_km": L / 1000, "head_elev_m": float(zf[head]),
                     "point_elev_m": float(zf[p]), "slope_m_per_m": S, "kirpich_tc_min": tc,
                     "dist_point_to_outlet_km": float(ddf[p]) / 1000})
    res_cells = np.flatnonzero(residual.ravel())
    rows.append({"element": "lateral_residual", "type": "screening_residual_support",
                 "area_km2": grid.area_km2(residual),
                 "dist_point_to_outlet_km": float(ddf[res_cells].mean()) / 1000 if len(res_cells) else 0.0,
                 "longest_flow_path_km": float(ddf[res_cells].max()) / 1000 if len(res_cells) else 0.0,
                 "kirpich_tc_min": np.nan})
    return rows


# ------------------------------------------------- chainage-anchored confluences (paper Table 15)
def main_stem_junctions(recv, stream, acc, dd, outlet):
    """Walk up the main stem (largest-accumulation donor) and list every junction with its D8 chainage.
    This reproduces the paper's check of named confluences against official chainage."""
    donors = _donors(recv, stream)
    ddf = dd.ravel()
    cur, out = int(outlet), []
    while True:
        ds = sorted(donors.get(cur, []), key=lambda d: acc[d], reverse=True)
        if not ds:
            break
        if len(ds) >= 2:
            out.append({"cell": cur, "chainage_km": float(ddf[cur]) / 1000.0, "main_donor": ds[0],
                        "tributary_donors": ds[1:], "trib_cells": [int(acc[d]) for d in ds[1:]]})
        cur = ds[0]
    return out


def anchored_group_points(junctions, anchors, tol_km, upstream_name, grid_cell_km2):
    """Place source-group points at named confluences using official chainage.
    anchors: [{name, chainage_km, graph_node?}] ordered from upstream to downstream is NOT required.
    Returns (points, report). `points` = [(upstream_name, main-stem cell just above the most upstream
    anchor), (anchor_name, tributary cell), ...]. `report` lists every candidate so a human can confirm
    that the D8 junction really is the named river (DSM + D8 can misplace channels)."""
    used, rep, pts = set(), [], []
    chosen = []
    for a in anchors:
        cands = sorted((j for j in junctions if j["cell"] not in used),
                       key=lambda j: abs(j["chainage_km"] - a["chainage_km"]))
        ok = [j for j in cands if abs(j["chainage_km"] - a["chainage_km"]) <= tol_km]
        near = [{"chainage_km": round(j["chainage_km"], 2), "trib_area_km2": round(j["trib_cells"][0] * grid_cell_km2, 1)}
                for j in cands[:3]]
        if not ok:
            rep.append({"anchor": a["name"], "official_chainage_km": a["chainage_km"], "found": None,
                        "status": f"NO main-stem junction within {tol_km} km", "nearest_candidates": near})
            continue
        j = ok[0]
        used.add(j["cell"])
        chosen.append((a, j))
        pts.append((a["name"], int(j["tributary_donors"][0])))
        rep.append({"anchor": a["name"], "graph_node": a.get("graph_node"), "official_chainage_km": a["chainage_km"],
                    "d8_chainage_km": round(j["chainage_km"], 2),
                    "chainage_error_km": round(j["chainage_km"] - a["chainage_km"], 2),
                    "tributary_area_km2": round(j["trib_cells"][0] * grid_cell_km2, 1),
                    "status": "CANDIDATE - confirm the D8 tributary is the named river against JICA/Phil-LiDAR hydrography",
                    "nearest_candidates": near})
    if chosen:
        up = max(chosen, key=lambda aj: aj[1]["chainage_km"])[1]
        pts.insert(0, (upstream_name, int(up["main_donor"])))
    return pts, rep
