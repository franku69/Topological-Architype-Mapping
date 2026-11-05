"""Abstract routing experiment on the six directed graphs (paper sections 3.11-3.19).

Equation numbers refer to the manuscript.  Time is an abstract tick, volume an abstract unit
(100 per event).  This is a RECONSTRUCTION from the written equations: the original code is lost,
so anything the manuscript does not specify (the stable hash for heterogeneous delays, which
scenario identifier is used for the random generator, which end of the score ranking is
'selected') is documented below and is a documented assumption, not a recovered fact.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import networkx as nx
import numpy as np
from scipy.signal import lfilter

from . import graphs

SEED = 260911
SEED_STRIDE = 104729
HORIZON = 512
EVENT = 100.0
CAPACITIES = (2, 5, 10, 20)
BUFFERS = (0, 10, 30)
GRID = dict(duration=(2, 8, 24), shape=("block", "triangle"), spread=(0, 8, 24), weights=("equal", "lognormal"),
            activation=("all", "subset"), delay=("unit", "hetero"), memory=(0, 2), realization=(0, 1, 2, 3))


# ------------------------------------------------------------------ forcing (Eq. 1)
def pulse(duration, shape):
    """Unit-sum pulse on `duration` ticks. Triangle weights min(i+1, d-i): for d=2 this equals the block,
    as the manuscript notes."""
    if shape == "block":
        w = np.ones(duration)
    else:
        w = np.array([min(i + 1, duration - i) for i in range(duration)], float)
    return w / w.sum()


def grid_settings():
    keys = list(GRID)
    return [dict(zip(keys, v), scenario=i) for i, v in enumerate(itertools.product(*GRID.values()))]


def source_draws(setting, n):
    """Common random numbers: six lognormal draws, six activation scores, six onset draws; first n used."""
    rng = np.random.default_rng(SEED + SEED_STRIDE * setting["scenario"])
    logn = rng.lognormal(0.0, 0.8, 6)[:n]
    score = rng.random(6)[:n]
    onset = rng.integers(0, setting["spread"] + 1, 6)[:n] if setting["spread"] > 0 else np.zeros(n, int)
    return logn, score, onset


def injections(setting, src_names, align=False):
    """dict source -> length-HORIZON array of injected volume. Weights renormalised after activation so the
    event is always 100 units."""
    n = len(src_names)
    logn, score, onset = source_draws(setting, n)
    w = np.ones(n) if setting["weights"] == "equal" else logn.copy()
    if setting.get("area_weights") is not None:
        w = np.asarray(setting["area_weights"], float)
    active = np.ones(n, bool)
    if setting["activation"] == "subset":
        k = math.ceil(0.6 * n)
        active[:] = False
        active[np.argsort(-score)[:k]] = True          # ASSUMPTION: highest fixed scores are selected
    w = np.where(active, w, 0.0)
    w = w / w.sum()
    p = pulse(setting["duration"], setting["shape"])
    out = {}
    for i, s in enumerate(src_names):
        a = 0 if align else int(onset[i])
        x = np.zeros(HORIZON)
        seg = (EVENT * w[i] * p)[: max(0, HORIZON - a)]
        x[a:a + len(seg)] = seg
        out[s] = x
    return out


# ------------------------------------------------------------------ routing (Eq. 2)
def linear_reservoir(x, tau):
    """Discrete linear reservoir: f = 1-exp(-1/tau); r_t = f (H_{t-1}+x_t); H_t = H_{t-1}+x_t-r_t."""
    if tau <= 0:
        return x
    f = 1.0 - math.exp(-1.0 / tau)
    H = lfilter([1 - f], [1, -(1 - f)], x)
    Hprev = np.concatenate(([0.0], H[:-1]))
    return x + Hprev - H


def shift(x, d):
    if d == 0:
        return x
    y = np.zeros_like(x)
    y[d:] = x[:-d]
    return y


def edge_delay(setting, origin):
    """Unit delay, or a seeded 1-4 delay from a stable hash of (seed, scenario, edge-origin key).
    ASSUMPTION: sha256 stands in for the original, unrecorded hash."""
    if setting["delay"] == "unit":
        return 1
    h = hashlib.sha256(f"{SEED}|{setting['scenario']}|{origin}".encode()).digest()
    return 1 + h[0] % 4


def route(G, setting, inj, tau_special=0.0, observe=None):
    """Forward routing in topological order with source tags kept.
    Returns {obs_node: {source: arrival series}} for each requested node (default: the receiver)."""
    observe = observe or [G.graph["receiver"]]
    order = list(nx.topological_sort(G))
    delays = {(u, v): (edge_delay(setting, u) if d["delay"] else 0) for u, v, d in G.edges(data=True)}
    res = {o: {} for o in observe}
    for s, x0 in inj.items():
        node_in = {n: np.zeros(HORIZON) for n in G.nodes}
        node_in[s] = node_in[s] + x0
        for n in order:
            if n != s and not nx.has_path(G, s, n):
                continue
            x = node_in[n]
            a = G.nodes[n]
            if n in res:
                res[n][s] = x.copy()
            if a["role"] == "storage":
                x = linear_reservoir(x, tau_special * a.get("storage_multiplier", 1.0))
            for v in G.successors(n):
                d = G.edges[n, v]
                y = x * d["frac"]
                if d["delay"]:
                    y = shift(linear_reservoir(y, float(setting["memory"])), delays[(n, v)])
                node_in[v] = node_in[v] + y
    return res


# ------------------------------------------------------------------ receiver (Eq. 3)
def queue(Q, capacities=CAPACITIES, buffers=BUFFERS):
    """Finite buffer, service-before-spill. Q: (S, T) arrivals.
    Returns total spill, shape (S, n_cap*n_buf), in abstract units (= % of the 100-unit event)."""
    cb = [(c, b) for c in capacities for b in buffers]
    C = np.array([c for c, _ in cb], float)[None, :]
    B = np.array([b for _, b in cb], float)[None, :]
    S = np.zeros((Q.shape[0], len(cb)))
    spill = np.zeros_like(S)
    for t in range(Q.shape[1]):
        A = S + Q[:, t:t + 1]
        Rt = np.minimum(C, A)
        F = np.maximum(0.0, A - Rt - B)
        S = A - Rt - F
        spill += F
    return spill, S


def synchrony(tagged):
    """Eq. 7: combined peak / sum of individual tagged peaks."""
    tot = sum(tagged.values())
    den = sum(v.max() for v in tagged.values())
    return float(tot.max() / den) if den > 0 else float("nan")


# ------------------------------------------------------------------ experiment drivers
def run_reference(G, observe_node=None, tau_special=0.0, settings=None, align=False, area_weights=None, G_override=None):
    """All grid settings -> arrival matrix and spill matrix at one observation node."""
    settings = settings or grid_settings()
    src = graphs.sources(G)
    obs = observe_node or G.graph["receiver"]
    Q, chi = [], []
    for st in settings:
        st = dict(st, area_weights=area_weights)
        tagged = route(G, st, injections(st, src, align=align), tau_special, [obs])[obs]
        tot = sum(tagged.values()) if tagged else np.zeros(HORIZON)
        Q.append(tot)
        chi.append(synchrony(tagged) if tagged else float("nan"))
    Q = np.array(Q)
    spill, final = queue(Q)
    mass = Q.sum(axis=1)[:, None] - (Q.sum(axis=1)[:, None] * 0)  # incoming
    return {"Q": Q, "spill": spill, "final_store": final, "chi": np.array(chi), "settings": settings}


def summarize(run):
    sp = run["spill"]
    return {"mean_spill_pct": float(sp.mean()),
            "any_spill_share_pct": float(100 * (sp > 1e-8).mean()),
            "median_spill_pct": float(np.median(sp)),
            "n_receiver_evaluations": int(sp.size)}


def paired_timing(G, observe_node=None, tau_special=0.0):
    """Strict paired timing (section 3.15): the 384 spread-24 settings, same everything but source starts
    aligned to tick 0. Returns mean change (pp) and higher/lower/tied shares over 384*12 = 4608 pairs."""
    sets = [s for s in grid_settings() if s["spread"] == 24]
    st = run_reference(G, observe_node, tau_special, sets, align=False)
    al = run_reference(G, observe_node, tau_special, sets, align=True)
    d = al["spill"] - st["spill"]
    tol = 1e-10
    return {"n_pairs": int(d.size), "mean_change_pp": float(d.mean()),
            "higher_pct": float(100 * (d > tol).mean()), "lower_pct": float(100 * (d < -tol).mean()),
            "tied_pct": float(100 * (np.abs(d) <= tol).mean())}


def conservation_error(run):
    """incoming = released + spilled + final retained, per setting (finite horizon)."""
    Q, sp, fin = run["Q"], run["spill"], run["final_store"]
    cb = [(c, b) for c in CAPACITIES for b in BUFFERS]
    err = 0.0
    for j, (c, b) in enumerate(cb):
        # released volume is recomputed by replaying the queue for this column
        S = np.zeros(Q.shape[0]); rel = np.zeros(Q.shape[0])
        for t in range(Q.shape[1]):
            A = S + Q[:, t]; R = np.minimum(c, A); F = np.maximum(0, A - R - b); S = A - R - F; rel += R
        err = max(err, float(np.abs(Q.sum(axis=1) - rel - sp[:, j] - S).max()))
    return err


# ------------------------------------------------------------------ arrangements (Eq. 6)
def n_binary_trees(n):
    """(2n-3)!! labelled rooted binary trees on n leaves."""
    return 1 if n < 2 else math.prod(range(1, 2 * n - 2, 2))


def enumerate_binary_trees(labels):
    """All rooted, unordered, binary trees on distinct labelled leaves (nested frozenset-free tuples,
    children sorted for a canonical form). Built by inserting each new leaf into any edge or above the root."""
    def canon(t):
        return t if isinstance(t, str) else tuple(sorted((canon(c) for c in t), key=repr))

    def inserts(t, leaf):
        yield canon((t, leaf))                                  # above this node
        if not isinstance(t, str):
            a, b = t
            for x in inserts(a, leaf):
                yield canon((x, b))
            for y in inserts(b, leaf):
                yield canon((a, y))

    trees = [canon((labels[0], labels[1]))]
    for leaf in labels[2:]:
        trees = list({x for t in trees for x in inserts(t, leaf)})
    return sorted(trees, key=repr)


def tree_shape(t):
    if isinstance(t, str):
        return "x"
    return "(" + "".join(sorted(tree_shape(c) for c in t)) + ")"


def arrangement_graph(tree, name="arr", receiver="R"):
    """Binary tree -> executed-graph style DAG: leaves are sources, internal nodes are mergers, the root is the
    receiver, plus a boundary node. All edges delay-bearing. Internal merger names are derived from their leaf set
    (canonical origin keys travel with the graph, as in section 3.16)."""
    G = nx.DiGraph(name=name, receiver=receiver)

    def leaves(t):
        return [t] if isinstance(t, str) else leaves(t[0]) + leaves(t[1])

    def add(t, is_root=False):
        if isinstance(t, str):
            G.add_node(t, role="source", label=t)
            return t
        nm = receiver if is_root else "M_" + "".join(sorted(leaves(t)))
        G.add_node(nm, role="receiver" if is_root else "junction", label=nm)
        for c in t:
            G.add_edge(add(c), nm, delay=True, frac=1.0)
        return nm
    add(tree, True)
    G.add_node("O", role="boundary", label="O")
    G.add_edge(receiver, "O", delay=True, frac=1.0)
    return G
