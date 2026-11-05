"""Core DEM hydrology in plain numpy (no GIS dependency, so it is unit-testable).

Pipeline:  DEM -> priority-flood conditioning (Barnes et al. 2014, epsilon variant)
           -> D8 receivers -> flow accumulation -> outlet snapping -> basin mask.

Every conditioning step is logged (how many cells were raised, by how much) so the
hydrological conditioning of a DSM can be reported in the paper (reviewer comment 1).
"""
from __future__ import annotations

import heapq
import numpy as np

# (drow, dcol) of the 8 D8 neighbours
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DR = np.array([d[0] for d in D8])
_DC = np.array([d[1] for d in D8])

M_LAT = 110574.0   # metres per degree latitude (approx.)
M_LON = 111320.0   # metres per degree longitude at the equator


class Grid:
    """North-up geographic (lon/lat degrees) grid -> metric step lengths and areas."""

    def __init__(self, lon0, lat0, res_x, res_y, shape):
        self.lon0, self.lat0, self.rx, self.ry = lon0, lat0, res_x, res_y
        self.shape = tuple(shape)
        rows = np.arange(self.shape[0])
        self.lat = lat0 - (rows + 0.5) * res_y
        self.dx = M_LON * np.cos(np.radians(self.lat)) * res_x   # metres per column, per row
        self.dy = M_LAT * res_y                                    # metres per row
        self.area = self.dx * self.dy                              # m2 per cell, per row

    @classmethod
    def from_transform(cls, t, shape):
        return cls(t.c, t.f, t.a, -t.e, shape)

    def rowcol(self, lon, lat):
        return int((self.lat0 - lat) / self.ry), int((lon - self.lon0) / self.rx)

    def lonlat(self, r, c):
        return self.lon0 + (c + 0.5) * self.rx, self.lat0 - (r + 0.5) * self.ry

    def area_km2(self, mask):
        """Area of a boolean mask in km2 (row-wise cos(lat) correction)."""
        return float((mask.sum(axis=1) * self.area).sum() / 1e6)


# --------------------------------------------------------------------------- conditioning
def priority_flood(dem, grid, sea_level=0.0, eps=1e-4, depression_threshold_m=0.1):
    """Fill depressions and resolve flats. Seeds = raster edge + cells <= sea_level.

    Returns (filled, order, seed, log). `order` is the flat-index pop order: every
    cell is popped after its downstream neighbour, so reversed(order) is a valid
    upstream->downstream sweep for accumulation.
    """
    raw = np.asarray(dem, dtype=np.float64)
    z = raw.copy()
    z[~np.isfinite(z)] = sea_level                     # nodata treated as sea (logged below)
    R, C = z.shape
    flat = z.ravel().tolist()
    filled = list(flat)
    closed = bytearray(R * C)
    seed = np.zeros((R, C), bool)
    seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
    seed |= z <= sea_level
    heap, n = [], 0
    for i in np.flatnonzero(seed.ravel()):
        i = int(i)
        closed[i] = 1
        heap.append((flat[i], n, i))
        n += 1
    heapq.heapify(heap)
    order = np.empty(R * C, dtype=np.int64)
    k = 0
    while heap:
        zc, _, i = heapq.heappop(heap)
        order[k] = i
        k += 1
        r, c = divmod(i, C)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C:
                j = rr * C + cc
                if not closed[j]:
                    closed[j] = 1
                    zj = flat[j]
                    if zj <= zc:
                        zj = zc + eps
                    filled[j] = zj
                    n += 1
                    heapq.heappush(heap, (zj, n, j))
    filled = np.array(filled).reshape(R, C)
    diff = filled - z
    depress = diff > depression_threshold_m
    log = {
        "algorithm": "priority-flood (Barnes et al. 2014), epsilon variant",
        "epsilon_m": eps,
        "sea_level_m": sea_level,
        "cells_total": int(R * C),
        "cells_raised_any": int((diff > 1e-9).sum()),
        "cells_raised_gt_threshold": int(depress.sum()),
        "depression_threshold_m": depression_threshold_m,
        "pct_cells_raised_gt_threshold": float(100 * depress.mean()),
        "max_raise_m": float(diff.max()),
        "fill_volume_m3": float((diff * depress * grid.area[:, None]).sum()),
        "nodata_cells_treated_as_sea": int((~np.isfinite(raw)).sum()),
    }
    return filled, order, seed, log


# --------------------------------------------------------------------------- D8 + accumulation
def flow_receivers(filled, seed, grid):
    """Steepest-descent D8 receiver (flat index, -1 = outlet/none)."""
    R, C = filled.shape
    pad = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros((R, C))
    best_k = np.full((R, C), -1)
    dxr = grid.dx[:, None]
    for k, (dr, dc) in enumerate(D8):
        nb = pad[1 + dr:1 + dr + R, 1 + dc:1 + dc + C]
        dist = np.hypot(dc * dxr, dr * grid.dy)
        slope = (filled - nb) / dist
        better = slope > best
        best = np.where(better, slope, best)
        best_k[better] = k
    recv = np.full(R * C, -1, dtype=np.int64)
    m = (best_k >= 0) & ~seed
    rows, cols = np.nonzero(m)
    kk = best_k[m]
    recv[rows * C + cols] = (rows + _DR[kk]) * C + (cols + _DC[kk])
    return recv


def accumulate(recv, order):
    """Flow accumulation in cells (each cell counts itself)."""
    r = recv.tolist()
    acc = [1] * len(r)
    for i in order[::-1].tolist():
        j = r[i]
        if j >= 0:
            acc[j] += acc[i]
    return np.array(acc, dtype=np.int64)


def distance_to_outlet(recv, order, grid, shape):
    """Along-flow-path distance (m) from every cell to its terminal outlet."""
    R, C = shape
    n = R * C
    idx = np.flatnonzero(recv >= 0)
    j = recv[idx]
    r, c = idx // C, idx % C
    rj, cj = j // C, j % C
    step_full = np.zeros(n)
    step_full[idx] = np.hypot((cj - c) * grid.dx[r], (rj - r) * grid.dy)
    rl, sl = recv.tolist(), step_full.tolist()
    dd = [0.0] * n
    for i in order.tolist():
        j2 = rl[i]
        if j2 >= 0:
            dd[i] = dd[j2] + sl[i]
    return np.array(dd)


def upstream_mask(recv, order, target, shape):
    """Boolean mask of all cells whose flow path passes through `target`."""
    rl = recv.tolist()
    m = bytearray(len(rl))
    for i in order.tolist():
        if i == target:
            m[i] = 1
        else:
            j = rl[i]
            if j >= 0 and m[j]:
                m[i] = 1
    return np.frombuffer(bytes(m), dtype=np.uint8).astype(bool).reshape(shape)


def snap_to_stream(acc, land, shape, row, col, radius_cells):
    """Move a point to the land cell with the largest accumulation within radius."""
    R, C = shape
    r0, r1 = max(0, row - radius_cells), min(R, row + radius_cells + 1)
    c0, c1 = max(0, col - radius_cells), min(C, col + radius_cells + 1)
    a = acc.reshape(R, C)[r0:r1, c0:c1].astype(float)
    a = np.where(land[r0:r1, c0:c1], a, -1)
    k = int(np.argmax(a))
    rr, cc = divmod(k, a.shape[1])
    return (r0 + rr) * C + (c0 + cc)


# --------------------------------------------------------------------------- terrain stats
def terrain_stats(z, grid, mask, relief_window_cells=11):
    """Elevation / slope / local-relief summaries inside `mask` (window or basin)."""
    from scipy import ndimage as ndi
    zz = np.where(np.isfinite(z), z, 0.0)
    gy, gx = np.gradient(zz, grid.dy, 1.0)
    gx = gx / grid.dx[:, None]
    slope_pct = 100 * np.hypot(gx, gy)
    relief = ndi.maximum_filter(zz, relief_window_cells) - ndi.minimum_filter(zz, relief_window_cells)

    def q(a, p):
        return float(np.percentile(a[mask], p))

    return {
        "area_km2": grid.area_km2(mask),
        "elev_min_m": float(zz[mask].min()), "elev_median_m": q(zz, 50),
        "elev_p90_m": q(zz, 90), "elev_max_m": float(zz[mask].max()),
        "slope_median_pct": q(slope_pct, 50), "slope_p90_pct": q(slope_pct, 90),
        "relief_median_m": q(relief, 50), "relief_p90_m": q(relief, 90),
        "relief_window_cells": relief_window_cells,
    }
