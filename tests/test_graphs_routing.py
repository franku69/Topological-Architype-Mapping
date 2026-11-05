"""Graph + routing tests. numpy/scipy/networkx only (no GIS libraries, no downloads)."""
import math
import numpy as np
from rtmap import graphs, routing as r

SPECS = graphs.load_specs(__import__("pathlib").Path(__file__).resolve().parents[1] / "config/graphs.yaml")


def test_all_six_graphs_match_table4():
    for k, s in SPECS.items():
        G = graphs.build(s)
        ok, msgs = graphs.validate(G, s)
        assert ok, (k, msgs)


def test_cdo_edges_exact():
    G = graphs.build(SPECS["cdo"])
    assert set(G.edges) == {("U", "J0"), ("T", "J0"), ("J0", "J1"), ("B", "J1"), ("J1", "R"), ("L", "R"), ("R", "O")}
    assert not any(a.get("is_delineated_catchment") for _, a in G.nodes(data=True))


def test_huong_urban_probe_excludes_bo_and_residual():
    G = graphs.build(SPECS["huong"])
    reach = graphs.passes_observation(G, "U")
    assert reach == {"T": True, "H": True, "B": False, "L": False}


def test_area_vectors_published():
    assert abs(sum(SPECS["kuranji"]["area_km2"].values()) - 202.69) < 1e-9
    assert abs(sum(SPECS["kadamaian"]["area_km2"].values()) - 359.36) < 1e-9
    assert SPECS["cdo"]["area_km2"] is None and SPECS["manado"]["area_km2"] is None and SPECS["hat_yai"]["area_km2"] is None


def test_pulse_and_forcing_sum_to_event():
    assert np.allclose(r.pulse(2, "block"), r.pulse(2, "triangle"))     # manuscript: 2-tick shapes coincide
    for d in (2, 8, 24):
        for sh in ("block", "triangle"):
            assert abs(r.pulse(d, sh).sum() - 1) < 1e-12
    for st in r.grid_settings()[::97]:
        inj = r.injections(st, list("UTBL"))
        assert abs(sum(x.sum() for x in inj.values()) - 100) < 1e-9       # also for subset activation


def test_grid_size():
    assert len(r.grid_settings()) == 1152
    assert len(r.CAPACITIES) * len(r.BUFFERS) == 12


def test_linear_reservoir_kernel():
    x = np.zeros(r.HORIZON); x[0] = 1.0
    y = r.linear_reservoir(x, 8.0)
    f = 1 - math.exp(-1 / 8)
    assert abs(y[0] - f) < 1e-12 and abs(y[1] - f * (1 - f)) < 1e-12 and abs(y.sum() - 1) < 1e-6


def test_queue_conservation_and_monotonicity():
    G = graphs.build(SPECS["cdo"])
    run = r.run_reference(G, settings=r.grid_settings()[::24])
    assert r.conservation_error(run) < 1e-9
    sp = run["spill"].reshape(len(run["Q"]), 4, 3)                     # (setting, capacity, buffer)
    assert (np.diff(sp, axis=1) <= 1e-9).all() and (np.diff(sp, axis=2) <= 1e-9).all()   # more resource never spills more


def test_synchrony_bounds():
    G = graphs.build(SPECS["cdo"])
    run = r.run_reference(G, settings=r.grid_settings()[::50])
    assert (run["chi"] > 0).all() and (run["chi"] <= 1 + 1e-12).all()


def test_total_mass_reaches_receiver():
    G = graphs.build(SPECS["kuranji"])
    st = dict(r.grid_settings()[0], area_weights=None)
    tag = r.route(G, st, r.injections(st, graphs.sources(G)))["R"]
    assert abs(sum(v.sum() for v in tag.values()) - 100) < 1e-6        # within the 512-tick horizon


def test_arrangement_counts():
    assert [r.n_binary_trees(n) for n in (2, 3, 4, 5)] == [1, 3, 15, 105]
    t4 = r.enumerate_binary_trees(list("UTBL"))
    assert len(t4) == 15 and len(r.enumerate_binary_trees(list("KPL"))) == 3 and len(r.enumerate_binary_trees(list("KBSJM"))) == 105
    shapes = [r.tree_shape(t) for t in t4]
    assert shapes.count("(((xx)x)x)") == 12 and shapes.count("((xx)(xx))") == 3    # 12 sequential, 3 balanced (paper 3.16)
    ref = r.arrangement_graph((((  "U", "T"), "B"), "L"))
    assert sorted(graphs.logical_distances(ref).values()) == [1, 2, 3, 3]   # reproduces the executed CDO shape


def test_paired_timing_counts():
    G = graphs.build(SPECS["cdo"])
    t = r.paired_timing(G)
    assert t["n_pairs"] == 4608 and t["mean_change_pp"] > 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
