"""Figures for the paper: window vs basin, streams, source groups."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def site_map(path, title, z, bbox, window, basin, reaches, groups, nodes, outlet_lonlat, provisional=False):
    fig, ax = plt.subplots(figsize=(8, 8 * (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]) + 0.6))
    ax.imshow(np.where(np.isfinite(z), z, np.nan), extent=(bbox[0], bbox[2], bbox[1], bbox[3]),
              cmap="Greys", alpha=0.55)
    window.boundary.plot(ax=ax, color="k", linestyle="--", linewidth=1, label="Analysis window (rectangle)")
    for i, (_, g) in enumerate(groups.iterrows()):
        c = "#bbbbbb" if g["group"] == "lateral_residual" else PALETTE[i % len(PALETTE)]
        gpd_row = groups.iloc[[i]]
        gpd_row.plot(ax=ax, color=c, alpha=0.35, edgecolor="none")
        ax.plot([], [], color=c, alpha=0.6, linewidth=8, label=f"{g['group']} ({g['area_km2']:.0f} km²)")
    basin.boundary.plot(ax=ax, color="#b30000", linewidth=1.6, label=f"Delineated basin ({basin['area_km2'].iloc[0]:.0f} km²)")
    for o, w in ((1, 0.5), (2, 0.9), (3, 1.4), (4, 2.0), (5, 2.8), (6, 3.5), (7, 4.2)):
        sub = reaches[reaches["strahler"] == o]
        if len(sub):
            sub.plot(ax=ax, color="#08519c", linewidth=w)
    junc = nodes[nodes["kind"] == "junction"]
    ax.scatter(junc["lon"], junc["lat"], s=4, c="k", zorder=5, label="Junctions")
    ax.scatter(*outlet_lonlat, marker="*", s=180, c="gold", edgecolor="k", zorder=6, label="Outlet")
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(title + ("  [PROVISIONAL CONFIG]" if provisional else ""), fontsize=11)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
