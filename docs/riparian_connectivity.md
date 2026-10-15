# Riparian Connectivity Analysis

## Overview

The `RiparianConnectivity` class in `hydrotools.morphology` provides an end-to-end
workflow for delineating three nested riparian zones around a stream network derived
from a Digital Elevation Model (DEM) and a mean annual precipitation grid:

| Zone | Name | Physical basis |
|------|------|----------------|
| 1 | Channel and active floodplain | Terrain within bankfull stage of the channel |
| 2 | Flood-prone area | Terrain within a multiple of bankfull depth (the flood-prone stage of Rosgen, 1994) |
| 3 | Groundwater-connected fringe | Terrain where a water table following topography remains within rooting depth of the surface |

Each zone is an *absolute*, physically interpretable extent rather than a relative
class. Zones are nested (1 ⊂ 2 ⊂ 3), contiguous with the channel, and scale with
stream size through bankfull depth and with channel gradient through a reach-scale
modifier. A continuous score grading connectivity within the zones is produced
alongside the classification.

The approach replaces the previous multi-criteria index (Topographic Wetness Index,
normalized attributes, and percentile classification). See
[Changes from version 0.1](#changes-from-version-01).

---

## Scientific Background

### Riparian Zones and Connectivity

Riparian zones are transitional areas between terrestrial and aquatic ecosystems whose
extent is governed by the interaction of fluvial geomorphology, hydrology, and
vegetation dynamics (Gregory et al., 1991; Naiman & Décamps, 1997). Lateral
connectivity between a channel and its margins operates through two principal
mechanisms:

1. **Surface inundation** — overbank flow reaches the floodplain when stage exceeds
   bankfull. The extent of inundation at a given stage is controlled by the height of
   the terrain above the channel.
2. **Groundwater exchange** — beyond the floodplain, riparian vegetation and wetlands
   are sustained where the water table remains close to the surface. In humid and
   temperate settings the water table is a subdued replica of topography (Tóth, 1963;
   Haitjema & Mitchell-Bruker, 2005), so the depth to water is governed by the
   elevation of the terrain above the stream and the hydraulic gradient.

Both mechanisms depend on the **rise** of the terrain above the channel. The workflow
therefore accumulates rise away from the stream network and compares it with
per-stream budgets that express each mechanism.

### Bankfull Width and Depth

Bankfull width is estimated with the empirical model of Hall et al. (2007):

$$
W_{bf} = C \cdot A^{\alpha} \cdot P^{\beta}
$$

where $A$ is contributing area (km²), $P$ is mean annual precipitation (cm/yr) and
$C = 0.042$, $\alpha = 0.48$, $\beta = 0.74$. Bankfull depth follows from width using
the companion relation from the same source:

$$
D_{bf} = 0.145 \cdot W_{bf}^{0.607}
$$

Because contributing area is computed from the DEM and precipitation is averaged over
the contributing area, both quantities are available at every stream cell. Mean annual
precipitation is therefore a **required** input. The default coefficients were fitted
in the Pacific Northwest; `morphology.fit_morphology_equation` can refit them where
observations are available.

### Rise Cost Surface

A friction surface is constructed from topographic slope:

$$
C_{ij} = \bar{s} \left( \tan\beta_{ij} + \varepsilon \right)
$$

where $\beta_{ij}$ is slope at cell $(i,j)$, $\bar{s}$ is the mean cell size and
$\varepsilon$ is a small gradient floor (default $\tan 0.5°$). GRASS `r.cost` (Dijkstra
with the knight's move) accumulates this friction outward from the stream network.
Because $\bar{s}\tan\beta$ is the elevation gained in crossing a cell, the cumulative
cost approximates the **metres of climb along the least-climb path from the stream** —
a path-based analogue of Height Above Nearest Drainage (Rennó et al., 2008; Nardi
et al., 2006) and the inverse of the downslope index of Hjerdt et al. (2004), which is
the standard DEM proxy for hydraulic gradient.

The floor $\varepsilon$ prevents unbounded spread across perfectly flat terrain
(filled sinks, reservoirs, terraces) and represents the decay of stream influence with
distance: on flat ground a budget of $T$ metres extends at most $T / \varepsilon$
metres laterally.

Channel cells carry the same friction as any other cell. This matters because stream
cells are seeded with different budgets (below): a larger stream's budget propagates
up a tributary only until the tributary channel has climbed the difference between the
two budgets — a backwater limit — after which the tributary's own budget prevails.

### Per-Stream Budgets

Each stream cell $i$ receives three nested rise budgets:

$$
\begin{aligned}
g_i  &= \text{slope modifier}(S_i) \\
T_{2,i} &= \max\left(f_2 \, D_{bf,i} \, g_i,\; t_{min} \tfrac{f_2}{f_1}\right) \\
T_{1,i} &= T_{2,i} \, f_1 / f_2 \\
T_{3,i} &= T_{2,i} + h_3
\end{aligned}
$$

A single `r.cost` run with the `-r` flag seeds each stream cell with an initial cost
$c_{0,i} = T_{max} - T_{3,i}$ where $T_{max} = \max_i T_{3,i}$. Cells with cumulative
cost below $T_{max}$ are within reach of *some* stream's budget, and Dijkstra
guarantees that the stream reaching a cell at least cost wins — so a small tributary
entering a large river's floodplain does not carve a notch out of it. The `nearest`
output of `r.cost` carries the winning seed value, from which the rise actually spent
and the winner's budgets are recovered per cell:

$$
\text{rise} = \text{cost} - c_0, \qquad
T_3 = T_{max} - c_0, \qquad
T_2 = T_3 - h_3, \qquad
T_1 = T_2 \, f_1 / f_2
$$

A cell belongs to the innermost zone whose budget exceeds its rise. The continuous
score is the fraction of the total budget remaining, $(T_3 - \text{rise}) / T_3$.
Channel cells are always zone 1 with a score of 1.

### Zone 1 and Zone 2: Floodplain Stages

$f_1$ and $f_2$ are multiples of bankfull depth. The defaults ($f_1 = 1$, $f_2 = 3$)
correspond to bankfull stage and to Rosgen's (1994) flood-prone stage, defined at
twice the *maximum* bankfull depth (roughly 2.5–3 times the mean depth estimated
here). The ratio of flood-prone width to bankfull width is the entrenchment ratio, so
zone 2 width divided by $W_{bf}$ along the network is directly comparable with
entrenchment and confinement classes (Rosgen, 1994; Fryirs et al., 2016; Beechie &
Imaki, 2014).

### Channel Gradient Modifier

Floodplain development is strongly conditioned by channel gradient: pool-riffle and
plane-bed reaches (gradient below ~1.5–3 %) build floodplains, whereas step-pool and
cascade reaches (above ~3–6.5 %) are confined with little lateral accommodation
(Montgomery & Buffington, 1997). The modifier $g$ is piecewise linear in channel
gradient $S$ (m/m):

$$
g(S) = \begin{cases}
g_{max} & S \le S_{low} \\
g_{max} - (g_{max} - g_{min}) \dfrac{S - S_{low}}{S_{high} - S_{low}} & S_{low} < S < S_{high} \\
g_{min} & S \ge S_{high}
\end{cases}
$$

with defaults $S_{low} = 0.015$, $S_{high} = 0.04$, $g_{min} = 0.5$, $g_{max} = 1.25$.
Channel gradient is computed on stream cells and averaged over a reach length
(default 200 m) so that it represents the reach rather than cell-scale noise. The
modifier scales the floodplain budgets ($T_1$, $T_2$) only; the groundwater band $h_3$
is unaffected because lateral gradient is already encoded in the cost surface.

The modifier is deliberately bounded. Steep reaches usually sit in confined valleys
where the lateral climb already consumes the budget; an unbounded modifier would
double-count that effect.

### Zone 3: Groundwater-Connected Fringe

Where the water table follows topography, its elevation near a gaining stream is
approximately

$$
z_{wt}(x) \approx z_{stream} + k \left( z_{surface}(x) - z_{stream} \right), \quad 0 < k < 1
$$

so the depth to water is $(1 - k) \cdot \text{rise}$. Vegetation and wetlands that
depend on groundwater require the water table within a rooting depth $d_r$ of the
surface, giving the condition

$$
\text{rise} < \frac{d_r}{1 - k} =: h_3
$$

Three properties follow:

- $h_3$ is a rise threshold, so zone 3 is produced by the same `r.cost` run as zones 1
  and 2.
- $h_3$ is **absolute** (metres), not scaled by stream size: rooting depth and the
  water-table ratio $k$ do not depend on channel dimensions. It is applied as a band
  above the flood-prone limit ($T_3 = T_2 + h_3$) so every stream retains a fringe.
- The lateral width of zone 3 is $h_3$ divided by the effective gradient — the
  "groundwater gradient proportional to slope" relationship made explicit.

The default $h_3 = 3$ m corresponds, for example, to $d_r \approx 1$ m and
$k \approx 0.67$. Reduce $h_3$ in arid settings, where the water table is decoupled
from topography (deep regional aquifers, karst, losing streams, heavy pumping), or
where only obligate wetland vegetation is of interest.

#### Convergence fringe (optional)

The rise threshold captures the *stream-stage* control on the water table. It misses
the *hillslope-supply* control: convergent hollows and toe slopes receive groundwater
in proportion to the area draining to them and remain connected at greater rise than
planar or divergent terrain — the mechanism behind the classical $\ln(a / \tan\beta)$
index (Beven & Kirkby, 1979). With `fringe_mode="convergence"` a second `r.cost -r`
pass is seeded from the floodplain (zones 1 and 2), carrying each cell's stream budget,
with friction multiplied by

$$
w(a) = \operatorname{clip}\left( \left( \frac{a_{hill}}{a} \right)^{\gamma},\; w_{min},\; 1 \right)
$$

where $a$ is contributing area, $a_{hill}$ (default one tenth of the stream
initiation area) is the area above which terrain is treated as convergent,
$\gamma = 0.5$ and $w_{min} = 0.25$. Cells reached within $h_3$ of weighted friction
from the floodplain edge are unioned into zone 3. Because $w \le 1$, zones 1 and 2 are
unchanged, the fringe is never smaller than in `"rise"` mode, and planar hillslopes
are unaffected. Contiguity with the floodplain is guaranteed by the seeding, which is
what allows upslope area to be used here without the artefacts that made it
unsuitable as a primary extent criterion.

### Assumptions and Limitations

- **Topography-driven water table.** Zone 3 assumes the water table is a subdued
  replica of topography. Where it is not, zone 3 overestimates groundwater connection.
- **DEM vertical resolution.** Zone 1 budgets are typically a few decimetres and are
  only resolved on high-accuracy (LiDAR) DEMs; on coarser products zones 1 and 2 may
  effectively merge. Zone 3 budgets (metres) are robust on most DEMs. `t_min`
  guarantees that zone 1 extends at least one cell beyond the channel.
- **Sink filling.** Flattened sinks and terraces reduce lateral friction to
  $\bar{s}\varepsilon$, so the lateral extent is bounded by $T / \varepsilon$ rather
  than by terrain.
- **Empirical hydraulic geometry.** Errors in $W_{bf}$ propagate to the floodplain
  budgets. Refit the coefficients regionally where possible.
- **Cost accumulation is direction-agnostic.** Descents are counted as climbs, so a
  levee or terrace riser blocks propagation in either direction. This is appropriate
  for inundation but means the surface should not be read as a hydraulic model.

---

## Workflow

```
DEM + annual precipitation
 │
 ├─► extract_streams()
 │       └── streams, flow direction, flow accumulation
 │
 ├─► calc_stream_attributes()
 │       ├── bankfull width  (contributing area, precipitation)
 │       ├── bankfull depth
 │       └── channel gradient (reach-averaged)
 │
 ├─► calc_zones(f1, f2, h3, slope_breaks, g_range, t_min, cost_floor, fringe_mode)
 │       ├── slope raster (derived internally)
 │       ├── per-stream budgets → seed raster (T_max − T3)
 │       ├── r.cost -r  → cumulative rise, nearest seed
 │       ├── [convergence] r.cost -r from the floodplain, area-weighted friction
 │       └── zones (1, 2, 3), score, region
 │
 └─► calc_connectivity(output_path, score_dst)
         └── zones as polygons or raster; optional score raster
```

Inspect `rc.zones` after `calc_zones()` and recalculate with different parameters as
needed; stream extraction and attributes do not need to be repeated.

---

## API Reference

### Constructor

#### `RiparianConnectivity(dem: str, annual_precip: str)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `dem` | `str` | DEM raster in a projected coordinate system with units of metres. |
| `annual_precip` | `str` | Mean annual precipitation grid (mm). Warped to the DEM grid with bilinear resampling if it does not align. |

Reads the DEM specifications (`csx`, `csy`, `top`, `bottom`, `left`, `right`,
`shape`, `wkt`) onto the instance, creates a temporary working directory, and
initializes all attribute slots to `None`.

### Step 1: Stream Extraction

#### `extract_streams(min_ws_area=1e6, min_stream_length=0, memory=4096)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_ws_area` | `float` | `1e6` | Minimum contributing area (m²) for stream initiation. |
| `min_stream_length` | `float` | `0` | Minimum stream segment length (m). |
| `memory` | `int \| None` | `4096` | Memory (MB) for GRASS `r.watershed`. |

Sets `streams`, `flow_direction` and `flow_accumulation`.

### Step 2: Stream Attributes

#### `calc_stream_attributes(focal_mean_dist=200.0)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `focal_mean_dist` | `float \| None` | `200.0` | Distance (m) over which channel gradient is averaged. `None` disables smoothing. |

Sets `bankfull_width` (m), `bankfull_depth` (m) and `stream_slope` (percent) on
stream cells. Requires `extract_streams()`.

### Step 3: Zone Delineation

#### `calc_zones(f1=1.0, f2=3.0, h3=3.0, slope_breaks=(0.015, 0.04), g_range=(0.5, 1.25), t_min=None, cost_floor=tan(0.5°), fringe_mode="rise", a_hill=None, gamma=0.5, w_min=0.25, memory=4096)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f1` | `float` | `1.0` | Bankfull depth multiple for zone 1 (bankfull stage). |
| `f2` | `float` | `3.0` | Bankfull depth multiple for zone 2 (flood-prone stage). Must exceed `f1`. |
| `h3` | `float` | `3.0` | Rise (m) above the zone 2 limit for zone 3; $d_r / (1 - k)$. |
| `slope_breaks` | `(float, float)` | `(0.015, 0.04)` | Channel gradients (m/m) at which the modifier reaches `g_max` and `g_min`. |
| `g_range` | `(float, float)` | `(0.5, 1.25)` | `(g_min, g_max)` bounds of the gradient modifier. |
| `t_min` | `float \| None` | `2 ε s̄` | Minimum zone 1 budget (m). Default is twice the friction of a flat cell. |
| `cost_floor` | `float` | `tan(0.5°)` | Gradient floor $\varepsilon$ (m/m). Lateral extent on flats is bounded by budget / `cost_floor`. |
| `fringe_mode` | `str` | `"rise"` | `"rise"` or `"convergence"` (adds the area-weighted second pass to zone 3). |
| `a_hill` | `float \| None` | `0.1 · min_ws_area` | Contributing area (m²) above which terrain is convergent (`"convergence"` only). |
| `gamma` | `float` | `0.5` | Rate of friction decrease with contributing area (`"convergence"` only). |
| `w_min` | `float` | `0.25` | Minimum friction weight (`"convergence"` only). |
| `memory` | `int \| None` | `4096` | Memory (MB) for GRASS `r.cost`. |

Sets `slope` (if not present), `start_cost`, `cost`, `nearest`, `zones`, `score`,
`region` and `t_max`; in `"convergence"` mode also `fringe_cost` and
`fringe_nearest`. Requires `calc_stream_attributes()`.

`r.cost` is run with `max_cost` set to `t_max`, so cells that cannot fall within any
zone are not visited.

### Step 4: Output

#### `calc_connectivity(connectivity_dst, score_dst=None, vector=True)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connectivity_dst` | `str` | — | Output path. Polygons (GeoPackage) when `vector=True`, otherwise a `uint8` raster with values 1–3 and nodata 0. |
| `score_dst` | `str \| None` | `None` | Optional raster with the fraction of rise budget remaining (0–1). |
| `vector` | `bool` | `True` | Write polygons rather than a raster. |

### Module-Level Functions

The class is a thin orchestration layer over three functions that can be used
independently:

#### `slope_modifier(stream_slope, slope_breaks=(0.015, 0.04), g_range=(0.5, 1.25)) -> dask.array`

Maps channel gradient (m/m) to the floodplain-width modifier $g$.

#### `rise_cost_surface(slope_src, start_cost_src, cost_dst, nearest_dst, cost_floor=tan(0.5°), max_cost=None, weight_src=None, knights_move=True, memory=4096)`

Builds the rise friction surface from a slope raster (degrees), optionally multiplied
by a weight raster, and runs `r.cost -r` seeded with `start_cost_src` (seed values on
stream cells, nodata elsewhere). Writes the cumulative cost and the nearest-seed
rasters.

#### `riparian_zones(cost_src, nearest_src, t_max, f1, f2, h3, zones_dst, score_dst=None, streams=None)`

Classifies the outputs of `rise_cost_surface` into zones 1–3 and, optionally, the
continuous score. When `streams` is supplied, channel cells are forced to zone 1.

#### `convergence_weight(flow_accumulation, cell_area, a_hill, gamma=0.5, w_min=0.25) -> dask.array`

Friction weights $w(a)$ from a contributing cell count.

#### `extend_fringe(fringe_cost_src, fringe_nearest_src, t_max, h3, zones_src, score_src, zones_dst, score_dst)`

Unions the convergence pass with first-pass zones and scores.

#### `region_width(region, streams, width_dst)`

Width of a boolean region measured across each stream cell (nearest-edge method).
Used by the validation script to measure zone 2 width along the network; also used by
`bankfull_width_geometric`.

### Utility Methods

#### `raster_path(name: str) -> str`

Path for a named raster inside the instance's temporary directory.

#### `verify_source(src: str)`

Raises `ValueError` if `src` does not align with the DEM grid.

---

## Complete Example

```python
from hydrotools.morphology import RiparianConnectivity

# 1. Initialize with a DEM (projected CRS, metres) and mean annual precipitation (mm)
rc = RiparianConnectivity("/data/dem.tif", "/data/annual_precip.tif")

# 2. Extract the stream network (1 km² minimum contributing area)
rc.extract_streams(min_ws_area=1e6)

# 3. Bankfull width/depth and reach-averaged channel gradient
rc.calc_stream_attributes(focal_mean_dist=200)

# 4. Delineate zones — inspect rc.zones and iterate on parameters as needed
rc.calc_zones(f1=1.0, f2=3.0, h3=3.0)

# 5. Write polygons and the continuous score
rc.calc_connectivity("/data/riparian_zones.gpkg", score_dst="/data/riparian_score.tif")
```

### Arid or decoupled-aquifer settings

```python
rc.calc_zones(h3=1.0)
```

### Emphasising gradient control on floodplain width

```python
rc.calc_zones(slope_breaks=(0.01, 0.03), g_range=(0.25, 1.5))
```

### Extending the fringe into convergent hollows

```python
rc.calc_zones(fringe_mode="convergence", a_hill=5e4, gamma=0.5, w_min=0.25)
```

### Raster output

```python
rc.calc_connectivity("/data/riparian_zones.tif", vector=False)
```

---

## Calibration and Validation

The three primary parameters each have an independent line of evidence:

| Parameter | Compare against |
|-----------|-----------------|
| `f1` | Mapped channel and active-floodplain polygons, LiDAR-derived channel extents. |
| `f2` | Mapped floodplains; the ratio of zone 2 width to $W_{bf}$ along the network against field or aerial estimates of entrenchment / confinement. |
| `h3` | Wetland inventories, riparian vegetation from land cover or late-season greenness, shallow well or seep observations. |

Zone 2 width along streams can be measured with `morphology.region_width`, and its
ratio to bankfull width should increase with contributing area in unconfined valleys
and remain near 1–2 in confined ones.

Because the zones are absolute, their areal proportions vary legitimately between
landscapes: a confined mountain catchment will be dominated by zone 3, a broad
alluvial valley by zones 1 and 2.

### Validation script

`scripts/validate_riparian_zones.py` runs the workflow and reports the diagnostics
above:

```
python scripts/validate_riparian_zones.py dem.tif precip.tif out/ \
    --floodplain-ref floodplain.tif --fringe-ref wetlands.tif \
    --sweep-f2 2 3 4 --sweep-h3 1 2 3 5
```

It writes the zones, score, streams and stream attributes to `out/`, together with:

- `stream_metrics.csv` — per stream cell: contributing area, bankfull width, channel
  gradient, zone 2 width and the width ratio; the Spearman correlation between area
  and zone 2 width and binned medians of the ratio are printed.
- Precision, recall, F1 and IoU of zones 1–2 against `--floodplain-ref`, and of zone
  3 and the full region against `--fringe-ref` (any non-zero raster, warped to the
  DEM grid if required).
- `sensitivity.csv` — zone areas and agreement metrics for each value in
  `--sweep-f2` / `--sweep-h3`.

Note that on coarse DEMs the ratio of zone 2 width to bankfull width is inflated for
small streams, because the minimum budget `t_min` and the cell size impose a floor on
zone width that is independent of channel size.

---

## Changes from version 0.1

Version 0.2 replaces the multi-criteria workflow. The following were removed:

- `calc_twi`, `define_region`, `calc_bankfull_width`, `calc_stream_slope`,
  `calc_stream_density`, `interpolate_into_region`
- The `attributes`, `quantize` and `percentiles` arguments of `calc_connectivity`

Rationale:

- Percentile classification forced one third of every study area into each class,
  making results relative to the sample and not comparable between areas or over time.
- The Topographic Wetness region used a single cost budget for every stream, so a
  headwater and a large river produced identical extents on the same hillslope.
- Bankfull width, stream slope and stream density are reach-scale properties; adding
  them to a per-cell score smeared them across the region. In the new model bankfull
  depth and channel gradient scale the zone budgets directly, and the role of stream
  density — connectivity from multiple nearby streams — is taken by the union of
  per-stream budgets.

The module-level functions `topographic_wetness`, `bankfull_width`, `bankfull_depth`,
`bankfull_extent`, `bankfull_width_geometric`, `valley_width` and
`valley_confinement` are unchanged.

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| GRASS GIS  | `r.watershed`, `r.stream.extract`, `r.cost` |
| GDAL       | `gdaldem slope`, raster I/O and warping |
| dask       | Lazy chunked array computation |
| rasterio   | Raster access |
| scikit-learn | `BallTree` nearest-edge queries in `region_width` |
| scipy      | `griddata` interpolation in `region_width`; statistics in the validation script |
| numpy      | Array operations |

---

## References

- Beechie, T. J., & Imaki, H. (2014). Predicting natural channel patterns based on
  landscape and geomorphic controls in the Columbia River basin, USA. *Water Resources
  Research*, 50(1), 39–57.

- Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable contributing
  area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43–69.

- Fryirs, K. A., Wheaton, J. M., & Brierley, G. J. (2016). An approach for measuring
  confinement and assessing the influence of valley setting on river forms and
  processes. *Earth Surface Processes and Landforms*, 41(5), 701–710.

- Gregory, S. V., Swanson, F. J., McKee, W. A., & Cummins, K. W. (1991). An
  ecosystem perspective of riparian zones. *BioScience*, 41(8), 540–551.

- Haitjema, H. M., & Mitchell-Bruker, S. (2005). Are water tables a subdued replica
  of the topography? *Ground Water*, 43(6), 781–786.

- Hall, J. E., Holzer, D. M., & Beechie, T. J. (2007). Predicting river floodplain
  and lateral channel migration for salmon habitat conservation. *Journal of the
  American Water Resources Association*, 43(3), 786–797.

- Hjerdt, K. N., McDonnell, J. J., Seibert, J., & Rodhe, A. (2004). A new
  topographic index to quantify downslope controls on local drainage. *Water Resources
  Research*, 40(5), W05602.

- Montgomery, D. R., & Buffington, J. M. (1997). Channel-reach morphology in mountain
  drainage basins. *Geological Society of America Bulletin*, 109(5), 596–611.

- Naiman, R. J., & Décamps, H. (1997). The ecology of interfaces: riparian zones.
  *Annual Review of Ecology and Systematics*, 28, 621–658.

- Nardi, F., Vivoni, E. R., & Grimaldi, S. (2006). Investigating a floodplain scaling
  relation using a hydrogeomorphic delineation method. *Water Resources Research*,
  42(9), W09409.

- Rennó, C. D., Nobre, A. D., Cuartas, L. A., Soares, J. V., Hodnett, M. G.,
  Tomasella, J., & Waterloo, M. J. (2008). HAND, a new terrain descriptor using
  SRTM-DEM: mapping terra-firme rainforest environments in Amazonia. *Remote Sensing
  of Environment*, 112(9), 3469–3481.

- Rosgen, D. L. (1994). A classification of natural rivers. *Catena*, 22(3), 169–199.

- Tóth, J. (1963). A theoretical analysis of groundwater flow in small drainage
  basins. *Journal of Geophysical Research*, 68(16), 4795–4812.
