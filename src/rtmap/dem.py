"""DEM acquisition: Copernicus GLO-90 (default) or any local DEM (FABDEM, LiDAR DTM, ...)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

GLO90_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/"
             "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
             "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")


def _tile_url(lat0: int, lon0: int) -> str:
    return GLO90_URL.format(ns="N" if lat0 >= 0 else "S", lat=abs(lat0),
                            ew="E" if lon0 >= 0 else "W", lon=abs(lon0))


def fetch_glo90(bbox, cache_dir):
    """bbox = (west, south, east, north). Downloads 1-degree GLO-90 tiles, mosaics, clips."""
    w, s, e, n = bbox
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    files = []
    for lat0 in range(math.floor(s), math.ceil(n)):
        for lon0 in range(math.floor(w), math.ceil(e)):
            fp = cache / Path(_tile_url(lat0, lon0)).name
            if not fp.exists():
                r = requests.get(_tile_url(lat0, lon0), timeout=120)
                if r.status_code == 404:        # open-ocean tile: no file exists
                    continue
                r.raise_for_status()
                fp.write_bytes(r.content)
            files.append(fp)
    if not files:
        raise RuntimeError("No GLO-90 tiles found for bbox")
    srcs = [rasterio.open(f) for f in files]
    arr, tr = merge(srcs, bounds=(w, s, e, n), nodata=np.nan)
    crs = srcs[0].crs
    for x in srcs:
        x.close()
    return arr[0].astype("float64"), tr, crs, {"source": "Copernicus GLO-90 DSM (3 arc-sec)",
                                                "tiles": [f.name for f in files]}


def load_local(path, bbox):
    """Clip any local DEM GeoTIFF to bbox (assumed EPSG:4326)."""
    from rasterio.windows import from_bounds
    with rasterio.open(path) as ds:
        win = from_bounds(*bbox, transform=ds.transform)
        arr = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype("float64")
        tr = ds.window_transform(win)
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        return arr, tr, ds.crs, {"source": str(path)}


def match_grid(path, transform, shape, crs):
    """Resample a local DEM onto the primary grid (for DSM-vs-DTM sensitivity runs)."""
    out = np.full(shape, np.nan, dtype="float64")
    with rasterio.open(path) as ds:
        reproject(rasterio.band(ds, 1), out, dst_transform=transform, dst_crs=crs,
                  src_nodata=ds.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def save_tif(path, arr, transform, crs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(arr.astype("float32"), 1)
