"""Sensitivity + audit against reference data (reviewer comments 1 and 3)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def mask_agreement(a, b, grid):
    inter, union = (a & b).sum(), (a | b).sum()
    return {"jaccard": float(inter / union) if union else None,
            "area_a_km2": grid.area_km2(a), "area_b_km2": grid.area_km2(b),
            "area_diff_pct": float(100 * (grid.area_km2(b) - grid.area_km2(a)) / max(grid.area_km2(a), 1e-9))}


def stream_agreement(a, b, tol_cells=1):
    """Share of stream cells in `a` with a stream cell of `b` within tol_cells (and vice versa)."""
    st = np.ones((3, 3), bool)
    bd = ndi.binary_dilation(b, st, iterations=tol_cells)
    ad = ndi.binary_dilation(a, st, iterations=tol_cells)
    return {"frac_a_within_tol_of_b": float((a & bd).sum() / max(a.sum(), 1)),
            "frac_b_within_tol_of_a": float((b & ad).sum() / max(b.sum(), 1)), "tol_cells": tol_cells}


def audit_reference(basin_gdf, streams_gdf, cfg_ref, base_dir):
    """Compare delineated basin / extracted streams against user-supplied reference layers
    (e.g. Phil-LiDAR / NAMRIA / agency watershed polygon and river lines)."""
    import geopandas as gpd
    out = {}
    if not cfg_ref:
        return out
    if cfg_ref.get("basin"):
        ref = gpd.read_file(f"{base_dir}/{cfg_ref['basin']}").to_crs(basin_gdf.crs)
        utm = basin_gdf.estimate_utm_crs()
        a, b = basin_gdf.to_crs(utm).unary_union, ref.to_crs(utm).unary_union
        out["basin_vs_reference"] = {"iou": a.intersection(b).area / a.union(b).area,
                                     "delineated_km2": a.area / 1e6, "reference_km2": b.area / 1e6,
                                     "area_diff_pct": 100 * (a.area - b.area) / b.area}
    if cfg_ref.get("streams"):
        ref = gpd.read_file(f"{base_dir}/{cfg_ref['streams']}").to_crs(streams_gdf.crs)
        utm = streams_gdf.estimate_utm_crs()
        tol = float(cfg_ref.get("tolerance_m", 200))
        s, r = streams_gdf.to_crs(utm).unary_union, ref.to_crs(utm).unary_union
        out["streams_vs_reference"] = {
            "tolerance_m": tol,
            "frac_extracted_length_within_tol_of_reference": s.intersection(r.buffer(tol)).length / s.length,
            "frac_reference_length_within_tol_of_extracted": r.intersection(s.buffer(tol)).length / r.length}
    return out
