#!/usr/bin/env python
"""Run the full mapping pipeline for one site (or all).

    python scripts/run_site.py cdo
    python scripts/run_site.py --all
"""
import argparse, hashlib, json, platform, sys, datetime as dt
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rtmap import hydro, network, dem as demmod, export, compare, maps, __version__  # noqa: E402


def process(z, grid, cfg, outlet_lonlat):
    """DEM array -> conditioned hydrology + basin. Returns a dict of arrays/objects."""
    shape = z.shape
    filled, order, seed, clog = hydro.priority_flood(z, grid, sea_level=cfg["sea_level_m"])
    recv = hydro.flow_receivers(filled, seed, grid)
    acc = hydro.accumulate(recv, order)
    land = np.nan_to_num(z, nan=0.0) > cfg["sea_level_m"]
    r, c = grid.rowcol(*outlet_lonlat)
    rad = max(1, int(round(cfg["snap_radius_m"] / grid.dy)))
    outlet = hydro.snap_to_stream(acc, land, shape, r, c, rad)
    member = hydro.upstream_mask(recv, order, outlet, shape)
    dd = hydro.distance_to_outlet(recv, order, grid, shape)
    return dict(filled=filled, order=order, recv=recv, acc=acc, land=land, outlet=outlet,
                member=member, dd=dd, cond_log=clog, shape=shape)


def run(site_key, sites, out_root):
    cfg = {**sites["defaults"], **sites["sites"][site_key]}
    out = Path(out_root) / site_key
    out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(cfg["bbox"])
    prov = cfg["status"] == "PROVISIONAL"
    if prov:
        print(f"[{site_key}] WARNING: PROVISIONAL bbox/outlet - confirm against your paper.")

    # 1) DEM --------------------------------------------------------------
    if cfg.get("dem_path"):
        z, tr, crs, dem_meta = demmod.load_local(ROOT / cfg["dem_path"], bbox)
    else:
        z, tr, crs, dem_meta = demmod.fetch_glo90(bbox, ROOT / "data/raw")
    demmod.save_tif(out / "dem.tif", z, tr, crs)
    shape = z.shape
    grid = hydro.Grid.from_transform(tr, shape)

    # 2) Conditioning, D8, basin delineation ------------------------------
    P = process(z, grid, cfg, tuple(cfg["outlet"]))
    member, outlet = P["member"], P["outlet"]
    C = shape[1]
    basin_km2 = grid.area_km2(member)
    on_edge = bool(member[0, :].any() or member[-1, :].any() or member[:, 0].any() or member[:, -1].any())
    warnings = []
    if on_edge:
        warnings.append("Delineated basin touches the DEM window edge: enlarge bbox (basin is truncated).")
    exp = cfg.get("expected")
    if exp and abs(basin_km2 - exp["basin_area_km2"]) / exp["basin_area_km2"] > exp["tol"]:
        warnings.append(f"Basin area {basin_km2:.1f} km2 outside expected {exp['basin_area_km2']} +/- {100*exp['tol']:.0f}% "
                        "- check outlet snapping / bbox / river identity.")
    for w in warnings:
        print(f"[{site_key}] WARNING:", w)

    # 3) Network, source groups, descriptors ------------------------------
    thr = int(round(cfg["stream_threshold_km2"] * 1e6 / float(np.mean(grid.area))))
    st = network.stream_mask(P["acc"], member, P["land"], thr)
    G = network.build_graph(P["recv"], st, P["acc"], P["filled"], grid, shape, outlet, P["dd"])
    desc = network.descriptors(G, basin_km2)
    sweep = {}
    for t_km2 in cfg["threshold_sweep_km2"]:
        tc = int(round(t_km2 * 1e6 / float(np.mean(grid.area))))
        s2 = network.stream_mask(P["acc"], member, P["land"], tc)
        G2 = network.build_graph(P["recv"], s2, P["acc"], P["filled"], grid, shape, outlet, P["dd"])
        d2 = network.descriptors(G2, basin_km2)
        sweep[str(t_km2)] = {k: d2[k] for k in ("n_sources", "n_junctions", "max_strahler_order",
                                                "drainage_density_km_per_km2", "source_to_outlet_km_cv")}

    anchor_report = None
    if cfg.get("source_group_anchors"):
        A = cfg["source_group_anchors"]
        junc = network.main_stem_junctions(P["recv"], st, P["acc"], P["dd"], outlet)
        pts, anchor_report = network.anchored_group_points(
            junc, A["anchors"], A.get("tolerance_km", 3.0), A["upstream_group"], float(np.mean(grid.area)) / 1e6)
        group_mode = "named confluences located by official chainage (candidates; confirm against hydrography)"
        for r in anchor_report:
            print(f"[{site_key}] anchor {r['anchor']}: {r.get('d8_chainage_km')} km (official {r['official_chainage_km']}) "
                  f"-> {r['status'][:40]}")
    elif cfg.get("source_groups"):
        pts = []
        for g in cfg["source_groups"]:
            rr, cc = grid.rowcol(g["lon"], g["lat"])
            rad = max(1, int(round(cfg["snap_radius_m"] / grid.dy)))
            pts.append((g["name"], hydro.snap_to_stream(P["acc"], P["land"], shape, rr, cc, rad)))
        group_mode = "named (user-supplied confluence points)"
    else:
        pts = network.auto_source_points(P["recv"], st, P["acc"], outlet, cfg["n_source_groups"])
        group_mode = "auto (largest tributaries joining the main stem)"
    masks, resid = network.delineate_groups(pts, P["recv"], P["order"], member, shape)
    table = network.group_table(pts, masks, resid, P["recv"], P["filled"], P["dd"], grid, shape, outlet)
    # map GIS support rows onto the conceptual graph nodes (graph node = conceptual; area = screening support)
    node_of = {}
    if cfg.get("source_group_anchors"):
        A = cfg["source_group_anchors"]
        node_of = {A["upstream_group"]: A.get("upstream_graph_node"), A.get("residual_group", "lateral_residual"): A.get("residual_graph_node")}
        node_of.update({a["name"]: a.get("graph_node") for a in A["anchors"]})
    for r in table:
        r["graph_node"] = (cfg.get("source_group_anchors", {}).get("residual_graph_node")
                           if r["element"] == "lateral_residual" else node_of.get(r["element"]))
        r["model_role"] = "GIS screening support only; the paper's graph node is a conceptual aggregate"

    # 4) Terrain stats: window (old method) vs basin (new method) ----------
    win_mask = np.isfinite(z)
    terrain = {"window_rectangle": hydro.terrain_stats(z, grid, win_mask),
               "delineated_basin": hydro.terrain_stats(z, grid, member)}

    # 5) Exports -----------------------------------------------------------
    win = export.window_gdf(bbox)
    basin = export.basin_gdf(member, tr, cfg["name"], basin_km2)
    reaches = export.reaches_gdf(G, grid, shape)
    nodes = export.nodes_gdf(G)
    groups = export.groups_gdf(masks, resid, tr, table)
    export.save(win, out / "analysis_window.geojson")
    export.save(basin, out / "basin.geojson")
    export.save(reaches, out / "streams.geojson")
    export.save(nodes, out / "nodes.geojson")
    export.save(groups, out / "source_groups.geojson")
    export.write_table(table, out / "subbasins_for_hms.csv")
    rt = [{"reach": f"{u}->{v}", **{k: d[k] for k in ("length_m", "drop_m", "slope", "strahler", "area_km2", "upstream_sources")}}
          for u, v, d in G.edges(data=True)]
    export.write_table(rt, out / "reaches_for_ras.csv")
    maps.site_map(out / "site_map.png", f"{cfg['name']} - basin vs analysis window", z, bbox, win, basin,
                  reaches, groups, nodes, grid.lonlat(outlet // C, outlet % C), prov)

    # 6) Sensitivity / audit ----------------------------------------------
    sens = {}
    if cfg.get("alt_dem_path"):
        z2 = demmod.match_grid(ROOT / cfg["alt_dem_path"], tr, shape, crs)
        P2 = process(z2, grid, cfg, tuple(cfg["outlet"]))
        st2 = network.stream_mask(P2["acc"], P2["member"], P2["land"], thr)
        sens = {"alt_dem": cfg["alt_dem_path"], "conditioning_primary": P["cond_log"],
                "conditioning_alt": P2["cond_log"], "basin": compare.mask_agreement(member, P2["member"], grid),
                "streams": compare.stream_agreement(st.reshape(shape), st2.reshape(shape))}
    audit = compare.audit_reference(basin, reaches, cfg.get("reference"), ROOT)

    # 7) Metrics + provenance ---------------------------------------------
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    metrics = {"site": site_key, "name": cfg["name"], "status": cfg["status"], "basin_area_km2": basin_km2,
               "outlet_snapped_lonlat": grid.lonlat(outlet // C, outlet % C), "source_group_mode": group_mode,
               "stream_threshold_km2": cfg["stream_threshold_km2"], "terrain": terrain,
               "descriptors": desc, "threshold_sweep": sweep, "source_groups": table,
               "confluence_anchor_report": anchor_report,
               "dem_sensitivity": sens, "reference_audit": audit, "warnings": warnings}
    export.write_json(metrics, out / "metrics.json")
    export.write_json(P["cond_log"], out / "conditioning_log.json")
    export.write_json({"rtmap_version": __version__, "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "python": platform.python_version(), "config_hash": cfg_hash, "config": cfg,
                       "dem": dem_meta, "dem_shape": list(shape),
                       "dem_resolution_deg": [tr.a, -tr.e]}, out / "provenance.json")
    print(f"[{site_key}] basin {basin_km2:.1f} km2 | groups: "
          + ", ".join(f"{r['element']} {r['area_km2']:.0f}" for r in table) + f" | outputs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "config/sites.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()
    sites = yaml.safe_load(open(a.config))
    keys = list(sites["sites"]) if a.all else [a.site]
    if not keys or keys == [None]:
        ap.error("give a site key or --all; available: " + ", ".join(sites["sites"]))
    for k in keys:
        run(k, sites, a.out)
