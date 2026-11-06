"""Synthetic-DEM sanity tests (numpy/networkx only; no GIS libs, no downloads)."""
import numpy as np
from rtmap import hydro, network


def make_dem(R=70, C=70):
    """Tilted plane to the south coast with two valleys that merge, plus one artificial pit."""
    r, c = np.mgrid[0:R, 0:C].astype(float)
    z = 400 * (1 - r / R) + 3
    z += 0.05 * np.hypot(c - C / 2, 0)                    # gentle cross-slope toward centre
    z -= 30 * np.exp(-((c - 20) ** 2) / 10)               # valley A
    z -= 30 * np.exp(-((c - 50) ** 2) / 10) * (r < R * 0.6)  # valley B joins later via cross cut
    z -= 40 * np.exp(-((r - 0.6 * R) ** 2) / 8) * ((c > 20) & (c < 50))  # cross valley
    z[30, 35] -= 25                                       # a pit to be filled
    z[-6:, :] = 0.0                                       # sea
    return z


def test_pipeline():
    z = make_dem()
    shape = z.shape
    grid = hydro.Grid(124.0, 8.0, 0.000833333, 0.000833333, shape)
    filled, order, seed, log = hydro.priority_flood(z, grid)
    assert log["cells_raised_gt_threshold"] >= 1 and log["max_raise_m"] > 10   # pit was filled
    recv = hydro.flow_receivers(filled, seed, grid)
    acc = hydro.accumulate(recv, order)
    land = z > 0
    # outlet = land cell with largest accumulation
    outlet = hydro.snap_to_stream(acc, land, shape, shape[0] - 8, shape[1] // 2, 40)
    member = hydro.upstream_mask(recv, order, outlet, shape)
    assert acc[outlet] == member.sum()                     # mass balance: acc == upstream cells
    dd = hydro.distance_to_outlet(recv, order, grid, shape)
    thr = 15
    st = network.stream_mask(acc, member, land, thr)
    G = network.build_graph(recv, st, acc, filled, grid, shape, outlet, dd)
    assert G.number_of_edges() == G.number_of_nodes() - 1  # a tree
    d = network.descriptors(G, grid.area_km2(member))
    assert d["n_sources"] >= 2 and d["max_strahler_order"] >= 2
    pts = network.auto_source_points(recv, st, acc, outlet, 3)
    masks, resid = network.delineate_groups(pts, recv, order, member, shape)
    tot = sum(m.sum() for m in masks.values()) + resid.sum()
    assert tot == member.sum()                             # groups + residual partition basin
    rows = network.group_table(pts, masks, resid, recv, filled, dd, grid, shape, outlet)
    assert abs(sum(r["area_km2"] for r in rows) - grid.area_km2(member)) < 1e-6


def test_anchors():
    """Chainage-anchored confluences: tributaries are found near their official chainage."""
    z = make_dem()
    grid = hydro.Grid(124.0, 8.0, 0.000833333, 0.000833333, z.shape)
    filled, order, seed, _ = hydro.priority_flood(z, grid)
    recv = hydro.flow_receivers(filled, seed, grid)
    acc = hydro.accumulate(recv, order)
    land = z > 0
    outlet = hydro.snap_to_stream(acc, land, z.shape, z.shape[0] - 8, z.shape[1] // 2, 40)
    member = hydro.upstream_mask(recv, order, outlet, z.shape)
    dd = hydro.distance_to_outlet(recv, order, grid, z.shape)
    st = network.stream_mask(acc, member, land, 15)
    J = network.main_stem_junctions(recv, st, acc, dd, outlet)
    assert J and all(j["chainage_km"] >= 0 for j in J)
    target = J[0]["chainage_km"]
    pts, rep = network.anchored_group_points(J, [{"name": "T", "chainage_km": target + 0.1, "graph_node": "T"}],
                                             1.0, "U", 0.0073)
    assert [n for n, _ in pts] == ["U", "T"] and abs(rep[0]["chainage_error_km"]) <= 1.0
    masks, resid = network.delineate_groups(pts, recv, order, member, z.shape)
    assert sum(m.sum() for m in masks.values()) + resid.sum() == member.sum()   # still a partition


def test_named_anchor():
    """Named-hydrography anchoring picks the tributary the line follows, not the one nearest a chainage."""
    z = make_dem()
    grid = hydro.Grid(124.0, 8.0, 0.000833333, 0.000833333, z.shape)
    filled, order, seed, _ = hydro.priority_flood(z, grid)
    recv = hydro.flow_receivers(filled, seed, grid)
    acc = hydro.accumulate(recv, order)
    land = z > 0
    outlet = hydro.snap_to_stream(acc, land, z.shape, z.shape[0] - 8, z.shape[1] // 2, 40)
    member = hydro.upstream_mask(recv, order, outlet, z.shape)
    dd = hydro.distance_to_outlet(recv, order, grid, z.shape)
    st = network.stream_mask(acc, member, land, 15)
    C = z.shape[1]
    mc = network.main_stem_cells(recv, st, acc, outlet)
    assert mc[0] == outlet and all(st[c] for c in mc)
    J = network.main_stem_junctions(recv, st, acc, dd, outlet)
    assert J
    # trace a "river line" up the D8 channel of the largest tributary of the first junction
    j = max(J, key=lambda j: j["trib_cells"][0])
    path, cur = [], int(j["tributary_donors"][0])
    donors = network._donors(recv, st)
    while cur is not None and len(path) < 12:
        path.append(cur)
        ds = donors.get(cur)
        cur = max(ds, key=lambda d: acc[d]) if ds else None
    line = np.array([grid.lonlat(c // C, c % C) for c in path[::-1]])
    # extend the line's downstream end onto the main stem (as OSM does: end ~11 m from the D8 stem)
    line = np.vstack([line, grid.lonlat(j["cell"] // C, j["cell"] % C)])
    rep = network.named_river_anchor("X", [line], recv, st, acc, dd, grid, z.shape, outlet, mc, 150.0)
    assert rep["status"].startswith("NAMED")
    assert rep["trib_cell"] == int(j["tributary_donors"][0])          # main-stem end cell was NOT chosen
    assert abs(rep["d8_chainage_km"] - j["chainage_km"]) < 0.01
    assert rep["osm_end_to_main_stem_m"] < 5.0
    # a wrong official chainage must not matter
    pts, out = network.named_group_points([rep], [{"name": "X", "chainage_km": j["chainage_km"] + 50}], "U", mc)
    assert [n for n, _ in pts] == ["U", "X"] and out[0]["chainage_error_km"] == -50.0
    masks, resid = network.delineate_groups(pts, recv, order, member, z.shape)
    assert sum(m.sum() for m in masks.values()) + resid.sum() == member.sum()
    # a line far from any channel is reported as not found
    far = np.array([[124.0, 8.0], [124.001, 8.001]])
    bad = network.named_river_anchor("Y", [far], recv, st, acc, dd, grid, z.shape, outlet, mc, 50.0)
    assert bad["status"].startswith("NOT FOUND") and "trib_cell" not in bad


if __name__ == "__main__":
    test_pipeline(); test_anchors(); test_named_anchor()
    print("synthetic pipeline test PASSED")
