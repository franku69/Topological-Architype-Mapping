#!/usr/bin/env python
"""Reviewer figure (comments 2 and 4): DEM window vs delineated basin, the four source groups, named confluences,
and a zoom on the Tumalaong mouth (OSM end vs D8 junction). Pure json + matplotlib (no GIS libraries).

    python scripts/make_reviewer_map.py cdo
"""
import argparse, json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
COL = {"U_upper_cdo": "#8fb3d9", "T_tumalaong": "#e8a87c", "B_bubunawan": "#9ccc9c", "lateral_residual": "#d9d9d9"}
LAB = {"U_upper_cdo": "U  upper CDO", "T_tumalaong": "T  Tumalaong", "B_bubunawan": "B  Bubunawan", "lateral_residual": "L  lateral residual (GIS remainder; no model polygon)"}


def load(p):
    return json.load(open(p))


def rings(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return polys


def patch(geom, **kw):
    verts, codes = [], []
    for poly in rings(geom):
        for ring in poly:
            ring = np.asarray(ring)[:, :2]
            verts += list(ring) + [ring[0]]
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 1) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw)


def main_stem(streams):
    """Edges of the main stem: from the outlet, follow the largest-area inflow upward."""
    into = {}
    for f in streams["features"]:
        p = f["properties"]
        into.setdefault(p["to_node"], []).append(f)
    tos = {f["properties"]["to_node"] for f in streams["features"]}
    froms = {f["properties"]["from_node"] for f in streams["features"]}
    cur = (tos - froms).pop()
    out = []
    while cur in into:
        e = max(into[cur], key=lambda f: f["properties"]["area_km2"])
        out.append(e)
        cur = e["properties"]["from_node"]
    return out


def hav_km(a, b):
    la = np.radians((a[1] + b[1]) / 2)
    return float(np.hypot((a[0] - b[0]) * 111.32 * np.cos(la), (a[1] - b[1]) * 110.574))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?", default="cdo")
    ap.add_argument("--osm", default=str(ROOT / "data/reference/cdo_named_rivers.geojson"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = ROOT / "outputs" / a.site
    cfg = yaml.safe_load(open(ROOT / "config/sites.yaml"))["sites"][a.site]
    m = load(od / "metrics.json")
    basin, groups, streams, nodes = (load(od / f"{n}.geojson") for n in ("basin", "source_groups", "streams", "nodes"))
    osm = load(a.osm) if Path(a.osm).exists() else {"features": []}
    W, S, E, N = cfg["bbox"]
    rows = {r["element"]: r for r in m["source_groups"]}
    ms = main_stem(streams)
    ms_xy = np.array([c for f in ms for c in f["geometry"]["coordinates"]])

    fig = plt.figure(figsize=(15, 8.2))
    ax = fig.add_axes([0.03, 0.06, 0.50, 0.84])
    zx = fig.add_axes([0.58, 0.06, 0.40, 0.84])
    for axx in (ax, zx):
        axx.set_aspect(1 / np.cos(np.radians(8.4)))
        axx.tick_params(labelsize=8)

    # ---- panel a: window vs basin vs groups
    ax.add_patch(Rectangle((W, S), E - W, N - S, fill=False, ls="--", lw=1.4, ec="#444", zorder=1))
    ax.text(W + 0.01, N - 0.02, f"DEM analysis window (rectangle)\n{m['terrain']['window_rectangle']['area_km2']:,.0f} km² - regional terrain context only",
            fontsize=9, va="top", color="#333")
    for g in groups["features"]:
        k = g["properties"]["group"]
        ax.add_patch(patch(g["geometry"], fc=COL.get(k, "#ccc"), ec="none", alpha=0.9, zorder=2))
    ax.add_patch(patch(basin["features"][0]["geometry"], fill=False, ec="#111", lw=1.6, zorder=5))
    for f in streams["features"]:
        c = np.array(f["geometry"]["coordinates"])
        ax.plot(c[:, 0], c[:, 1], color="#2b6cb0", lw=0.5 + 0.35 * f["properties"]["strahler"], alpha=0.9, zorder=4)
    for f in osm["features"]:
        c = np.array(f["geometry"]["coordinates"])
        ax.plot(c[:, 0], c[:, 1], color="#b00020", lw=1.0, zorder=6)
    ol, oa = m["outlet_snapped_lonlat"]
    ax.plot(ol, oa, "k^", ms=9, zorder=8)
    ax.annotate("CDO mouth", (ol, oa), xytext=(8, 6), textcoords="offset points", fontsize=9, weight="bold")
    for k, mk in (("T_tumalaong", "o"), ("B_bubunawan", "s")):
        r = rows[k]
        ax.plot(r["confluence_lon"], r["confluence_lat"], mk, ms=9, mfc="white", mec="k", mew=1.6, zorder=9)
        ax.annotate(k[0], (r["confluence_lon"], r["confluence_lat"]), xytext=(9, -3), textcoords="offset points",
                    fontsize=12, weight="bold", zorder=9)
    # group labels at area-weighted centroids of polygon vertices
    for g in groups["features"]:
        k = g["properties"]["group"]
        pts = np.vstack([np.asarray(r)[:, :2] for poly in rings(g["geometry"]) for r in poly[:1]])
        cx, cy = pts.mean(0)
        txt = f"{LAB[k].split()[0]}\n{g['properties']['area_km2']:.0f} km²"
        box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        if k in ("T_tumalaong", "B_bubunawan"):
            # narrow strips: put the label in the empty east side of the window with a leader line
            ring = np.vstack([np.asarray(r)[:, :2] for poly in rings(g["geometry"]) for r in poly[:1]])
            pp = MPath(np.vstack([np.vstack([np.asarray(r)[:, :2], np.asarray(r)[:1, :2]]) for poly in rings(g["geometry"]) for r in poly[:1]]))
            gx, gy = np.meshgrid(np.linspace(ring[:, 0].min(), ring[:, 0].max(), 220), np.linspace(ring[:, 1].min(), ring[:, 1].max(), 220))
            gp = np.column_stack([gx.ravel(), gy.ravel()])
            inside = gp[pp.contains_points(gp)]
            # interior point nearest the middle of the polygon's interior samples (a point guaranteed inside)
            tgt = inside[np.argmin(np.hypot(inside[:, 0] - inside[:, 0].mean(), inside[:, 1] - inside[:, 1].mean()))]
            pos = (E - 0.12, 8.17) if k == "B_bubunawan" else (E - 0.12, 8.06)
            ax.annotate(txt, tgt, xytext=pos, ha="center", va="center", fontsize=11, weight="bold", zorder=10, bbox=box,
                        arrowprops=dict(arrowstyle="-", color="k", lw=0.9, shrinkB=0))
        else:
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=11, weight="bold", zorder=10, bbox=box)
    ax.set_xlim(W - 0.01, E + 0.01); ax.set_ylim(S - 0.01, N + 0.01)
    ax.set_title(f"(a) CDO basin ({m['basin_area_km2']:,.0f} km²) inside the DEM window, with the four source groups", fontsize=11, loc="left")
    ax.set_xlabel("Longitude (°E)", fontsize=9); ax.set_ylabel("Latitude (°N)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[k], ec="none") for k in COL]
    handles += [plt.Line2D([], [], color="#2b6cb0", lw=1.5), plt.Line2D([], [], color="#b00020", lw=1.2),
                plt.Line2D([], [], color="k", lw=1.6), plt.Line2D([], [], marker="o", ls="", mfc="w", mec="k"),
                plt.Line2D([], [], marker="s", ls="", mfc="w", mec="k")]
    ax.legend(handles, [LAB[k] for k in COL] + ["D8 streams (>=10 km²)", "OSM Tumalaong / Bubunawan", "Delineated basin",
              "Tumalaong junction (D8)", "Bubunawan junction (D8)"], loc="upper right", fontsize=7.5, framealpha=0.95)

    # ---- panel b: Tumalaong mouth
    t = rows["T_tumalaong"]; b = rows["B_bubunawan"]
    tum = [np.array(f["geometry"]["coordinates"]) for f in osm["features"] if f["properties"].get("name", "").startswith("Tumalaong")]
    gap_txt = ""
    tj = (t["confluence_lon"], t["confluence_lat"])
    xs = [tj[0], b["confluence_lon"]]; ys = [tj[1], b["confluence_lat"]]
    if tum:
        ends = np.vstack([np.vstack([c[0], c[-1]]) for c in tum])
        dm = np.hypot((ends[:, None, 0] - ms_xy[None, :, 0]) * 111.32 * np.cos(np.radians(8.4)),
                      (ends[:, None, 1] - ms_xy[None, :, 1]) * 110.574)
        i, j = np.unravel_index(dm.argmin(), dm.shape)
        oe = ends[i]; oe_ms = ms_xy[j]
        straight = hav_km(oe, tj)
        zx.plot(*oe, marker="*", ms=16, mfc="#b00020", mec="k", zorder=12)
        zx.annotate(f"OSM Tumalaong ends\n{dm[i, j]*1000:.0f} m from D8 main stem", oe, xytext=(-150, 40), textcoords="offset points",
                    fontsize=9, arrowprops=dict(arrowstyle="-", color="k"), zorder=12)
        xs.append(oe[0]); ys.append(oe[1])
        gap_txt = f"OSM end to D8 junction: {straight:.1f} km straight-line"
    zx.plot(*tj, "o", ms=11, mfc="white", mec="k", mew=1.8, zorder=12)
    zx.annotate("T: D8 junction\n(pipeline anchor)", tj, xytext=(-120, 22), textcoords="offset points", fontsize=9, zorder=12,
                arrowprops=dict(arrowstyle="-", color="k", lw=0.8), bbox=dict(fc="white", ec="none", alpha=0.85))
    zx.plot(b["confluence_lon"], b["confluence_lat"], "s", ms=10, mfc="white", mec="k", mew=1.8, zorder=12)
    zx.annotate("B: Bubunawan junction", (b["confluence_lon"], b["confluence_lat"]), xytext=(10, 8), textcoords="offset points", fontsize=9, zorder=12)
    for g in groups["features"]:
        zx.add_patch(patch(g["geometry"], fc=COL[g["properties"]["group"]], ec="none", alpha=0.55, zorder=1))
    for f in streams["features"]:
        c = np.array(f["geometry"]["coordinates"]); zx.plot(c[:, 0], c[:, 1], color="#2b6cb0", lw=1.0 + 0.4 * f["properties"]["strahler"], zorder=4)
    for f in osm["features"]:
        c = np.array(f["geometry"]["coordinates"]); zx.plot(c[:, 0], c[:, 1], color="#b00020", lw=1.6, zorder=6)
    pad = 0.035
    zx.set_xlim(min(xs) - pad, max(xs) + pad); zx.set_ylim(min(ys) - pad, max(ys) + pad)
    zx.set_title("(b) Zoom: Tumalaong mouth - OSM river end vs D8 junction", fontsize=11, loc="left")
    rep = {r.get("anchor"): r for r in (m.get("confluence_anchor_report") or [])}.get("T_tumalaong", {})
    cap = gap_txt
    if rep.get("osm_end_nearest_main_stem_chainage_km") is not None:
        cap += (f"\nMain-stem chainage: OSM end {rep['osm_end_nearest_main_stem_chainage_km']:.1f} km | "
                f"D8 junction {rep['d8_chainage_km']:.1f} km | JICA CH {rep.get('official_chainage_km')} km")
    zx.text(0.02, 0.02, cap, transform=zx.transAxes, fontsize=8.5, va="bottom", bbox=dict(fc="white", ec="#888", alpha=0.92))
    zx.set_xlabel("Longitude (°E)", fontsize=9)

    fig.suptitle("Cagayan de Oro River - GIS screening support for the conceptual source groups U, T, B, L "
                 "(screening layer only; the model's groups are conceptual, see Fig. 5a)", fontsize=11.5, y=0.995)
    out = Path(a.out) if a.out else od / "reviewer_map.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
