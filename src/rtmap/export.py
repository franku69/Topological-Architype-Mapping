"""GeoJSON / table exports (the files a GIS reviewer can open directly)."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import LineString, Point, box, shape as shp_shape
from shapely.ops import unary_union


def mask_to_polygon(mask, transform):
    geoms = [shp_shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=mask, transform=transform) if v == 1]
    return unary_union(geoms) if geoms else None


def save(gdf, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def window_gdf(bbox):
    return gpd.GeoDataFrame({"name": ["analysis_window"]}, geometry=[box(*bbox)], crs=4326)


def basin_gdf(mask, transform, name, area_km2):
    return gpd.GeoDataFrame({"name": [name], "area_km2": [area_km2]},
                            geometry=[mask_to_polygon(mask, transform)], crs=4326)


def groups_gdf(masks, residual, transform, table):
    rows, geoms = [], []
    tab = {r["element"]: r for r in table}
    for n, m in {**masks, "lateral_residual": residual}.items():
        if m.any():
            rows.append({"group": n, "area_km2": tab[n]["area_km2"], "kind": tab[n]["type"]})
            geoms.append(mask_to_polygon(m, transform))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)


def reaches_gdf(G, grid, shape):
    C = shape[1]
    rows, geoms = [], []
    for u, v, d in G.edges(data=True):
        pts = [grid.lonlat(i // C, i % C) for i in d["cells"]]
        if len(pts) < 2:
            continue
        rows.append({"from_node": u, "to_node": v, "length_m": d["length_m"], "drop_m": d["drop_m"],
                     "slope": d["slope"], "strahler": d["strahler"], "area_km2": d["area_km2"],
                     "upstream_sources": d["upstream_sources"]})
        geoms.append(LineString(pts))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)


def nodes_gdf(G):
    rows = [{"node": n, **{k: v for k, v in d.items()}} for n, d in G.nodes(data=True)]
    return gpd.GeoDataFrame(rows, geometry=[Point(r["lon"], r["lat"]) for r in rows], crs=4326)


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def write_table(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
