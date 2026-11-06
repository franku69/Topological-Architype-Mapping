#!/usr/bin/env python
"""Download named OSM waterway lines for a site's bbox -> data/reference/<site>_named_rivers.geojson.

    python scripts/fetch_osm_rivers.py cdo
    python scripts/fetch_osm_rivers.py cdo --names "Tumalaong|Bubunawan"

Uses the Overpass API (needs internet). Names default to the `river_match` values in
config/sites.yaml -> source_group_anchors.anchors. Output is GeoJSON LineStrings with `name`, `osm_id`, `waterway`.
Note: data/reference/* is git-ignored, so commit this file explicitly (git add -f) if you want it versioned.
"""
import argparse, json, sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
URL = "https://overpass-api.de/api/interpreter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--names", help="regex of river names (default: from config anchors)")
    ap.add_argument("--config", default=str(ROOT / "config/sites.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["sites"][a.site]
    names = a.names or "|".join(x["river_match"] for x in cfg["source_group_anchors"]["anchors"])
    w, s, e, n = cfg["bbox"]
    q = f'[out:json][timeout:90];way["waterway"~"river|stream|canal"]["name"~"{names}",i]({s},{w},{n},{e});out geom;'
    r = requests.post(URL, data={"data": q}, timeout=120,
                      headers={"User-Agent": "rtmap-topological-mapping/0.1 (academic flood-topology study)",
                               "Accept": "application/json"})
    r.raise_for_status()
    feats = [{"type": "Feature",
              "properties": {"name": el["tags"].get("name"), "osm_id": el["id"], "waterway": el["tags"].get("waterway")},
              "geometry": {"type": "LineString", "coordinates": [[g["lon"], g["lat"]] for g in el["geometry"]]}}
             for el in r.json()["elements"] if el.get("geometry")]
    out = ROOT / cfg["source_group_anchors"]["hydrography"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"wrote {len(feats)} ways -> {out}")
    for nm in sorted({f['properties']['name'] for f in feats}):
        print("  ", nm, sum(f['properties']['name'] == nm for f in feats), "ways")
    if not feats:
        sys.exit("no ways found - check --names / bbox")


if __name__ == "__main__":
    main()
