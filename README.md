# river-topology-mapping

Rebuilt repository for *"Hydro-topological patterns and residual downstream overflow in Cagayan de Oro
and five Southeast Asian river systems"*.

The repo keeps **three different spatial objects separate**, because that separation is the answer to
most of the GIS reviewer's questions:

| Object | What it is | Where | Used by the paper's routing experiment? |
|---|---|---|---|
| **Graph** (U, T, B, L, J0, J1, R, O ...) | Directed acyclic graph of named river components and declared aggregates. Source groups are **conceptual aggregations, not delineated catchments**. | `config/graphs.yaml`, `src/rtmap/graphs.py` | **Yes** (the only spatial input) |
| **Terrain window** | The rectangular DEM crop (CDO: 124.55-125.08 E, 7.99-8.56 N). Regional context only. | `config/sites.yaml -> bbox` | No |
| **GIS screening layer** | Conditioned D8 network, basin delineated from the outlet, candidate catchments upstream of named confluences. Evidence for the reviewer, **not** an authoritative delineation. | `scripts/run_site.py` | No |

## Reviewer comments -> what the repo does

| # | Comment | Answer in the repo | Where |
|---|---|---|---|
| 1 | DSM vs DTM; conditioning | Priority-flood conditioning with a full log (cells raised, max raise, fill volume, NoData). Optional `alt_dem_path` re-runs everything on a DTM (FABDEM / Phil-LiDAR) and reports basin overlap and stream agreement. The original preprocessing log was never archived; this is a **new, separately documented terrain run**, not a recovery. | `outputs/<site>/conditioning_log.json`, `metrics.json -> dem_sensitivity` |
| 2 | Rectangle vs CDO basin | Statistics are written for the rectangle **and** for the basin delineated from the outlet, side by side, plus a map showing both. A warning fires if the basin touches the window edge. | `analysis_window.geojson`, `basin.geojson`, `site_map.png`, `metrics.json -> terrain` |
| 3 | Higher-resolution DEM / Phil-LiDAR | Put reference layers in `data/reference/` and point to them in `sites.yaml` (`reference:`, `alt_dem_path:`); the repo reports basin IoU and % stream length within a tolerance. No such layer is bundled; none was used in the paper. | `metrics.json -> reference_audit` |
| 4 | Are source groups catchments? | **No - conceptual aggregates** (paper 3.3, Table 4a/4b). The graph encodes them as nodes; `nodes.csv` has `is_delineated_catchment = False` for every node. For the reviewer's map, optional GIS **support** polygons are produced: catchments upstream of the Tumalaong (JICA CH 22.3) and Bubunawan (CH 18.5) confluences, found by chainage along the D8 main stem, with the remainder reported as `lateral_residual`. They are labelled `screening_catchment_support`. Unequal aggregation across the six rivers is visible in `descriptors.json` and in Table 4a of the paper. | `config/graphs.yaml`, `outputs/graphs/<site>/`, `outputs/cdo/source_groups.geojson`, `metrics.json -> confluence_anchor_report` |
| 5 | Link to HEC-HMS / HEC-RAS | Exports subbasin areas, longest flow path, slope, Kirpich Tc (screening) and confluence coordinates, plus a reach table (length, drop, slope, Strahler, area). These are screening hand-offs for choosing where to put surveys and subbasin boundaries, **not** calibrated parameters. | `subbasins_for_hms.csv`, `reaches_for_ras.csv` |

## Run

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python scripts/run_graphs.py         # no DEM needed: builds and validates the six graphs vs Table 4
python scripts/run_experiments.py    # no DEM needed: the paper's routing experiment, vs reported values
python scripts/run_site.py cdo       # GIS layer (needs internet once: Copernicus GLO-90 from AWS, or set dem_path)
python tests/test_synthetic.py && python tests/test_graphs_routing.py     # (pytest tests/ also works)
```

## Graph layer (`graphs.yaml`, `graphs.py`)
Edge lists were transcribed from Figures 5, 16, 29, 43, 58, 72 and checked automatically against Table 4:
node count, edge count, source count, storage positions and logical distances all match for all six graphs
(`outputs/graphs/validation.csv`). `evidence:` blocks contain only what the manuscript states; map view
URLs, capture dates and digitising confidence are `null` because the manuscript says they were not recorded.
**Please fill them in** (reviewer comment RR2 asks for exactly this).

Two things were inferred rather than read: (a) the Huong source-to-reservoir connectors are zero-delay
(the only assignment that reproduces Table 4 distances 4,4,2,2; the figure draws no dashes), and (b) arrow
direction in Hat Yai/Huong splits is 0.5/0.5 as stated. Confirm (a).

## Routing experiment (`routing.py`) - how close is the reconstruction?
Implements sections 3.11-3.19: 100-unit normalised event (Eq. 1), 1152-setting grid, linear-reservoir
memory (Eq. 2), finite buffer with service-before-spill (Eq. 3), synchrony chi (Eq. 7), strict paired
timing, and enumeration of labelled binary arrangements (Eq. 6: 3, 15, 105). Results against the manuscript:

| Site | Mean spill % (rebuilt / paper) | Any-spill % (rebuilt / paper) | Timing pp (rebuilt / paper) |
|---|---|---|---|
| CDO | 18.19 / 18.39 | 49.42 / 49.88 | 7.25 / 7.49 |
| Manado | 21.96 / 22.05 | 54.32 / 55.04 | 7.25 / 7.96 |
| Kuranji | 20.09 / 20.33 | 51.09 / 52.21 | 12.48 / 12.82 |
| Huong | 17.04 / 17.18 | 46.93 / 47.21 | 6.42 / 6.76 |
| Kadamaian | 21.79 / 22.00 | 54.06 / 54.90 | 7.10 / 7.66 |
| Hat Yai | 17.79 / 17.95 | 47.98 / 48.47 | 12.66 / 13.21 |

Counts match exactly (13,824 receiver evaluations and 4,608 timing pairs per case); conservation error is
~1e-13. Values are **close but not identical**: the manuscript does not specify the hash behind the
heterogeneous 1-4 tick delays, which scenario identifier seeds the generator, or which end of the score
ranking is "selected", so these are documented assumptions in `routing.py`. **Do not cite these rebuilt
numbers as the paper's results; the paper's archived results stay authoritative.** Huong urban probe
5.45 vs 5.46 reported; Hat Yai urban 20.81 vs 21.01.

Not re-implemented: storage sweeps, the Huong position alternatives and Kuranji refinement/area-weight runs
are supported by the building blocks (`storage_multiplier`, `alternatives`, `area_km2`,
`enumerate_binary_trees`) but not run end to end here.

## GIS layer notes and limits
* Default DEM Copernicus GLO-90 (a **DSM**); D8 on a DSM is a screening network - bends are shortened and
  bridges/vegetation can misplace channels. Lake Tondano is a flat surface routed artificially.
* Snapping the outlet picks the highest-accumulation land cell within `snap_radius_m`.
* Only CDO (window and outlet 8.4971 N, 124.6521 E) and Kuranji have bounds taken from your reports; Tondano uses
  the JICA frame; **Hue, Kota Belud and Hat Yai are PROVISIONAL**. The CDO `expected` basin area
  (1,139.08 km2, in-window outlet group) is the regression target.
* The chainage finder returns *candidates* with D8 chainage error and tributary area; a human must confirm
  that the D8 tributary is the named river.

## Suggested methods paragraph
"Source groups in the six executed graphs are conceptual aggregations constrained by named river
relationships and official chainage; they are not delineated catchments (Table 4a). The rectangular DEM
window provides regional terrain context only. For GIS review, the Copernicus GLO-90 DSM was conditioned with
priority-flood, a basin was delineated from the river-mouth outlet with D8, and candidate catchments were
placed at the Tumalaong and Bubunawan confluences by matching D8 streams to named OSM river lines (official chainages reported alongside D8 chainages for audit). These layers are screening
evidence and do not enter the routing experiment."


## Re-running CDO with named-river anchoring
```
pip install -r requirements.txt
python scripts/fetch_osm_rivers.py cdo        # writes data/reference/cdo_named_rivers.geojson (needs internet)
python scripts/run_site.py cdo                # compare outputs/cdo/metrics.json -> confluence_anchor_report
PYTHONPATH=src python tests/test_synthetic.py
```
`confluence_anchor_report` lists, per river: D8 chainage vs official chainage, tributary area, median offset between
the D8 channel and the OSM line, and how far the OSM downstream end sits from the D8 main stem. If the hydrography
file is absent the run falls back to chainage matching and `source_group_mode` says `FALLBACK`.
