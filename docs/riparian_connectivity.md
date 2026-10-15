# Riparian Connectivity Analysis

## Overview

The `RiparianConnectivity` class in `hydrotools.morphology` provides an end-to-end
workflow for delineating and classifying riparian zones. Starting from a Digital
Elevation Model (DEM), the workflow estimates channel size (bankfull width) and
gradient (stream slope) along a derived stream network. It converts them into a
**riparian reach** in metres at every stream cell, then expands that reach across the
terrain with a least-cost (`r.cost`) analysis. The reach left over at each cell is a
continuous connectivity index. It is used both as the riparian extent and as the value
that gets classified into connectivity zones, exported as a raster or vector dataset.

The approach is grounded in the principle that riparian connectivity — the degree of
lateral hydrological and ecological linkage between a stream and its floodplain — is
controlled by channel size, channel gradient, and the topographic resistance between the
channel and its surroundings (Naiman & Décamps, 1997; Gregory et al., 1991).

---

## Scientific Background

### Riparian Zones and Connectivity

Riparian zones are transitional areas between terrestrial and aquatic ecosystems. Their
extent and condition are governed by the interaction of fluvial geomorphology,
hydrology, and vegetation dynamics. Quantifying riparian connectivity is important for
habitat assessment, restoration prioritization, and land-use planning (Naiman et al.,
2005).

### Bankfull Width Estimation

The empirical bankfull width model implemented here follows the general form:

$$
W_{bf} = C \cdot A^{\alpha} \cdot P^{\beta}
$$

Where:
- $W_{bf}$ is bankfull width (m)
- $A$ is contributing area (km²)
- $P$ is mean annual precipitation (cm/yr)
- $C$, $\alpha$, $\beta$ are empirically fitted coefficients

Default coefficients ($C = 0.042$, $\alpha = 0.48$, $\beta = 0.74$) are derived from:

> Hall, J. E., Holzer, D. M., & Beechie, T. J. (2007). Predicting river floodplain
> and lateral channel migration for salmon habitat conservation. *Journal of the
> American Water Resources Association*, 43(3), 786–797.
> https://doi.org/10.1111/j.1752-1688.2007.00063.x

This power-law relationship between drainage area, precipitation, and channel geometry
is well-established in fluvial geomorphology (Leopold & Maddock, 1953; Dunne & Leopold,
1978).

### Least-Cost Riparian Expansion

Riparian extent and connectivity are modelled with a single **cost-distance** surface
from streams, computed with GRASS GIS `r.cost`. Unlike a classical cost-distance where
every stream cell starts at zero, each stream cell is seeded with an initial cost that
reflects its own channel size and gradient (the `-r` flag of `r.cost`). The terrain cost
surface is in the same units (metres) as the seeded values, so the expansion across the
floodplain carries on the bankfull-width magnitudes instead of switching to a separate,
unitless index.

#### Why cost-distance rather than classical TWI?

The classical TWI ($\ln(a / \tan\beta)$) quantifies the propensity of a location to
accumulate water based on upslope contributing area ($a$) and local slope
($\tan\beta$). While effective for hillslope hydrology, it has several limitations
when the goal is to delineate **lateral riparian connectivity**:

1. **Sensitivity to DEM artefacts.** The upslope contributing area term is highly
   sensitive to small errors in flow routing, especially in flat valley bottoms where
   flow direction is ambiguous (Sørensen et al., 2006; Kopecký & Čížková, 2010).

2. **Non-directional with respect to streams.** Classical TWI describes a cell's
   tendency to be wet, but does not explicitly encode *distance or cost to the nearest
   stream channel*. Two cells with identical TWI values may differ greatly in their
   lateral accessibility from a stream.

3. **Poor discrimination in low-relief terrain.** In broad valley floors, upslope area
   is uniformly large and slope uniformly low, producing a plateau of high TWI values
   with little internal structure (Grabs et al., 2009).

Cost-distance analysis addresses these limitations by computing the **minimum
cumulative friction** required to travel from the nearest stream cell to every
other cell in the landscape, using slope as the friction surface. This approach:

- Explicitly models **lateral accessibility from the channel network**, which is the
  physical basis of riparian connectivity (Tockner & Stanford, 2002; Jencso et al.,
  2009).
- Produces smooth, well-differentiated surfaces in valley bottoms because cost
  accumulates monotonically with distance, even across flat terrain.
- Naturally integrates both **proximity to streams** and **topographic resistance**
  (slope) into a single surface, capturing the concept that riparian influence
  attenuates with both distance and terrain steepness.

#### Precedent in the literature

Cost-distance from streams has been used as a riparian delineation tool and
connectivity proxy in several contexts:

- **Riparian zone mapping.** Abood et al. (2012) used cost-distance with slope as a
  friction surface to delineate variable-width riparian buffers, demonstrating that
  topographic cost-distance better captures the functional extent of riparian zones
  than fixed-width buffers, particularly in mountainous terrain.

- **Floodplain and valley-bottom delineation.** Gallant & Dowling (2003) developed the
  Multi-Resolution Valley Bottom Flatness (MrVBF) index using slope-based criteria to
  identify valley floors. While MrVBF uses a different algorithm, the underlying
  rationale — that valley bottoms are characterized by low slope and proximity to
  drainage — is shared by the cost-distance approach. Gilbert et al. (2016) extended
  cost-distance methods specifically for valley-bottom mapping, finding that
  slope-weighted distance from streams produced delineations comparable to manual
  field-verified mapping.

- **Lateral hydrologic connectivity.** Jencso et al. (2009) showed that the degree of
  hillslope-riparian-stream connectivity is governed by topographic convergence and
  the planform geometry relating hillslopes to valley floors — properties that a
  cost-distance surface captures directly. Tockner & Stanford (2002) framed
  floodplain connectivity as a function of lateral distance modulated by topographic
  and hydraulic resistance, which is precisely what a slope-weighted cost surface
  encodes.

- **Cost surfaces in ecology and conservation.** The use of friction-weighted distance
  (rather than Euclidean distance) to model connectivity is well established in
  landscape ecology (Adriaensen et al., 2003; McRae et al., 2008). The same
  principles apply when the "habitat patches" are stream channels and the
  "landscape matrix" is the surrounding terrain.

#### Riparian reach at streams

At each stream cell, a riparian reach $R$ (m) is defined as a multiple of bankfull
width, attenuated by stream slope:

$$
R = k_w \cdot W_{bf} \cdot \left(1 - \frac{\min(S, S_c)}{S_c}\right)
$$

Where:
- $W_{bf}$ is bankfull width (m)
- $S$ is stream slope (degrees)
- $S_c$ is a slope cutoff (default: 15°), at which the reach is reduced to zero
- $k_w$ is a width factor (default: 3), the reach on each side of the channel as a
  multiple of bankfull width

$R$ is capped at a fixed maximum reach $R_{\max}$ (default: 100 m). Larger, lower-gradient channels get a larger reach. This follows the
positive relationship between channel size and floodplain extent (Leopold & Maddock,
1953; Hall et al., 2007) and the inverse relationship between channel gradient and
floodplain width (Montgomery & Buffington, 1997).

#### Initial cost (`r.cost -r`)

`r.cost` accumulates cost upwards from its starting values, so the reach is converted to
an initial cost relative to a fixed budget $B = R_{\max}$:

$$
C^{0} = B - R
$$

A stream at the maximum reach starts at zero cost. A stream with no reach starts at the
budget, so its riparian zone does not extend past the channel.

The budget is a parameter, not a property of the study area (such as the largest reach
present). Every term is therefore absolute: bankfull width (m), stream slope (degrees),
the effective distance cost (m), and the budget (m). The same channel and terrain
produce the same value in any watershed, so results are **comparable between
watersheds** analyzed with the same parameters.

#### Terrain cost surface

The cost of crossing a cell is an **effective distance** in metres: the horizontal
distance travelled plus the vertical distance climbed, weighted by $k_v$:

$$
c_{ij} = \bar{s} \cdot \left(1 + k_v \tan(\theta_{ij})\right)
$$

Where $\theta_{ij}$ is the topographic slope at cell $(i,j)$, $\bar{s}$ is the average
cell size (`r.cost` measures moves in cell units), and $k_v$ is the vertical weight
(default: 20), the effective horizontal metres charged for each metre of climb. On flat
valley floors the cost is close to plain distance, so the riparian zone extends roughly
$R$ metres from the channel. On valley walls the cost rises quickly and the expansion
stops.

#### Riparian region and connectivity

The accumulated cost $C_{ij}$ (Dijkstra's shortest path with the knight's move) is
limited to the budget (`max_cost=B`), and the remaining reach is normalized:

$$
\text{region}_{ij} = \frac{B - C_{ij}}{B} \quad \text{where } C_{ij} \leq B
$$

Cells with $C_{ij} > B$ are outside the riparian region. Values range from 1 (a stream
at the maximum reach) to 0 (the riparian boundary). Higher values mean the
cell is closer, with less climbing, to a larger, lower-gradient channel. Multiplying by
$B$ gives the remaining reach in metres. Where riparian zones from several streams
overlap, the cell takes the minimum accumulated cost, so a large river's riparian zone
naturally envelops the mouths of its tributaries.

The knight's move extends the 8-connected neighbourhood to include cells reachable by a
chess knight's L-shaped move, improving angular resolution of the cost surface and
reducing directional bias inherent in raster-based distance computations (GRASS GIS
Reference Manual).

### Stream Slope

Stream slope is derived by masking the DEM to only stream-cell locations and computing
slope (via GDAL `gdaldem slope`) on the masked surface. This yields local longitudinal
gradient at each stream pixel. An optional focal-mean smoothing step averages slope
values over a specified neighbourhood, producing segment-representative gradients rather
than cell-level noise. Any stream cell without a stream slope value falls back to the
terrain slope.

### Classification

The continuous region surface is optionally discretized into ordinal zones. By default,
**fixed breaks** on the region value (0–1) are used, which keeps the classes comparable
between watersheds. The default breaks divide the surface into three classes:

| Class | Region value | Remaining reach (default $R_{\max}$ = 100 m) | Interpretation |
|-------|--------------|-----------------------------------------------|----------------|
| 0     | $< 1/3$      | $< 33$ m      | Low connectivity |
| 1     | $1/3$ – $2/3$ | $33$ – $67$ m | Moderate connectivity |
| 2     | $> 2/3$      | $> 67$ m      | High connectivity |

Percentile thresholds computed from the distribution of non-masked values may be used
instead. Each class then holds a set share of the riparian area (e.g. one-third with
terciles), which is useful when absolute thresholds are unknown (Jenks, 1967; Eastman,
1999). Percentile classes are relative to the study area, though, so they are **not
comparable between watersheds**.

The classified result is vectorized with corner-smoothing to produce polygon regions
suitable for overlay analysis, habitat assessment, or restoration planning in standard
GIS software.

---

## Workflow

```
DEM
 │
 ├─► extract_streams()
 │       │
 │       ├── streams raster
 │       └── flow direction raster
 │
 ├─► calc_bankfull_width(annual_precip)
 │       │
 │       └── bankfull width (m) along streams
 │
 ├─► calc_stream_slope()
 │       │
 │       └── stream slope (degrees) along streams
 │
 ├─► define_region(width_factor, slope_cutoff, vertical_weight)
 │       │
 │       ├── slope raster (derived internally)
 │       ├── riparian reach at streams → r.cost initial cost (-r)
 │       ├── effective distance cost surface
 │       └── normalized remaining reach (riparian region + connectivity)
 │
 └─► calc_connectivity(output_path)
         │
         └── classification of the region (fixed breaks) → vector/raster
```

---

## API Reference

### Constructor

#### `RiparianConnectivity(dem: str)`

Initialize the analysis with a DEM. The DEM must be in a **projected coordinate
system with units in metres**.

**Parameters:**
| Parameter | Type  | Description |
|-----------|-------|-------------|
| `dem`     | `str` | Path to a DEM raster (any GDAL-supported format). |

**Behaviour:**
- Reads raster metadata (cell size, extent, projection, shape) from the DEM.
- Creates a temporary directory for intermediate rasters.
- Initializes all attribute slots (`slope`, `flow_direction`, `streams`,
  `bankfull_width`, `stream_slope`, `region`, `reach_budget`) to `None`.

**Key attributes set on the instance:**
| Attribute            | Type          | Description |
|----------------------|---------------|-------------|
| `dem`                | `str`         | Path to the input DEM. |
| `csx`, `csy`         | `float`       | Cell size in x and y directions (m). |
| `top`, `bottom`, `left`, `right` | `float` | Raster extent. |
| `shape`              | `tuple`       | Array shape `(bands, rows, cols)`. |
| `wkt`                | `str`         | Well-Known Text projection string. |
| `tmp_dir`            | `TemporaryDirectory` | Managed temp directory for intermediates. |

---

### Step 1: Stream Extraction

#### `extract_streams(min_ws_area=1e6, min_stream_length=0, memory=4096)`

Derive a stream network from the DEM using flow accumulation thresholding.

**Parameters:**
| Parameter           | Type               | Default | Description |
|---------------------|--------------------|---------|-------------|
| `min_ws_area`       | `float`            | `1e6`   | Minimum contributing area in m² to classify a cell as a stream. 1e6 = 1 km². |
| `min_stream_length` | `float`            | `0`     | Minimum stream segment length in metres. Segments shorter than this are removed. |
| `memory`            | `Union[int, None]` | `4096`  | Memory allocation (MB) for GRASS `r.watershed`. |

**Behaviour:**
1. If `flow_accumulation` has not been computed, runs `flow_direction_accumulation()`
   (GRASS `r.watershed`) to produce flow direction and accumulation grids.
2. Runs `extract_streams()` (GRASS `r.stream.extract`) to derive a stream raster and
   flow direction raster from the accumulation surface.
3. Sets `self.streams` and `self.flow_accumulation`.

**Dependencies:** GRASS GIS with `r.watershed` and `r.stream.extract`.

**Scientific note:** The contributing area threshold controls stream initiation. The
default of 1 km² is a common choice in meso-scale hydrological modelling (O'Callaghan
& Mark, 1984; Tarboton et al., 1991).

---

### Step 2: Bankfull Width

#### `calc_bankfull_width(annual_precip_src: str)`

Compute empirical bankfull width (m) along streams.

**Parameters:**
| Parameter           | Type  | Default | Description |
|---------------------|-------|---------|-------------|
| `annual_precip_src` | `str` | —       | Path to a mean annual precipitation raster in mm. Must align spatially with the DEM. |

**Behaviour:**
1. Validates spatial alignment of the precipitation raster with the study area.
2. Computes contributing area (km²) and mean annual precipitation (cm/yr) along streams
   using flow accumulation routing (`FlowAccumulation.contributing_area` and
   `FlowAccumulation.calculate` with `method="mean"`).
3. Applies the Hall et al. (2007) equation:
   $W_{bf} = 0.042 \cdot A^{0.48} \cdot P^{0.74}$
4. Sets `self.bankfull_width`.

**Prerequisite:** `extract_streams()`.

---

### Step 3: Stream Slope

#### `calc_stream_slope(focal_mean_dist=None)`

Compute longitudinal slope (degrees) along stream cells.

**Parameters:**
| Parameter         | Type                 | Default | Description |
|-------------------|----------------------|---------|-------------|
| `focal_mean_dist` | `Union[None, float]` | `None`  | Distance (m) of a focal mean used to smooth stream slope. |

**Behaviour:**
1. Masks the DEM to stream-cell locations only.
2. Computes slope on the masked DEM using GDAL `gdaldem slope`, optionally smoothed.
3. Sets `self.stream_slope`.

**Prerequisite:** `extract_streams()`.

---

### Step 4: Riparian Region Definition

#### `define_region(width_factor=3.0, slope_cutoff=15.0, vertical_weight=20.0, max_reach=100.0, knights_move=True, memory=4096)`

Delineate the riparian region and its continuous connectivity with a least-cost
expansion from streams.

**Parameters:**
| Parameter         | Type                 | Default | Description |
|-------------------|----------------------|---------|-------------|
| `width_factor`    | `float`              | `3.0`   | Riparian reach on each side of a stream as a multiple of bankfull width ($k_w$). |
| `slope_cutoff`    | `float`              | `15.0`  | Stream slope (degrees) at which the reach is reduced to zero ($S_c$). |
| `vertical_weight` | `float`              | `20.0`  | Effective horizontal metres charged per metre of vertical climb ($k_v$). |
| `max_reach`       | `float`              | `100.0` | Maximum riparian reach (m), used as the fixed budget $B$. Keep constant between watersheds that will be compared. |
| `knights_move`    | `bool`               | `True`  | Use the knight's move in `r.cost`. |
| `memory`          | `Union[int, None]`   | `4096`  | Memory allocation (MB) for GRASS `r.cost`. |

**Behaviour:**
1. Derives topographic slope from the DEM if it has not been computed.
2. Computes the riparian reach $R$ at each stream cell, capped at $B$ = `max_reach`.
3. Writes the initial cost $B - R$ at stream cells and the effective distance cost
   surface.
4. Runs GRASS `r.cost` with the `-r` flag (and `-k` for the knight's move), limited to
   `max_cost=B`.
5. Saves $(B - C) / B$ as the region raster and sets `self.region` and
   `self.reach_budget` ($B$, in metres).

**Iterative calibration:** This method is designed to be called repeatedly. After each
call, inspect `self.region` in a GIS viewer to verify the riparian extent aligns with
observed valley bottoms. Increase `width_factor` to widen riparian zones everywhere,
and increase `vertical_weight` to confine them more tightly to flat valley floors.

**Prerequisites:** `calc_bankfull_width()` and `calc_stream_slope()`.

---

### Step 5: Connectivity Calculation

#### `calc_connectivity(connectivity_dst, quantize=True, breaks=[1/3, 2/3], percentiles=None, vector=True)`

Classify the riparian region into connectivity zones.

**Parameters:**
| Parameter          | Type   | Default | Description |
|--------------------|--------|---------|-------------|
| `connectivity_dst` | `str`  | —       | Output file path. Vector (GeoPackage) if `vector=True`, raster (GeoTIFF) otherwise. |
| `quantize`         | `bool` | `True`  | Whether to classify the continuous region into discrete zones. |
| `breaks`           | `list` | `[1/3, 2/3]` | Fixed region values (0–1) for classification, comparable between watersheds. |
| `percentiles`      | `Union[None, list]` | `None` | Percentile boundaries used in place of `breaks`. Relative to the study area. |
| `vector`           | `bool` | `True`  | Export as vector (requires `quantize=True`). |

**Behaviour:**
1. Loads the region raster produced by `define_region()`.
2. If `quantize=True`, applies `np.digitize` with `breaks` (or with thresholds computed
   from `percentiles`) to assign each cell an ordinal class (0 = low connectivity).
3. Writes a smoothed polygon GeoPackage if `vector=True`, otherwise a Cloud-Optimized
   GeoTIFF (continuous or classified).

**Prerequisite:** `define_region()`.

**Classification notes:**
- `breaks` must be monotonically increasing and within 0–1. `percentiles` must be
  monotonically increasing and within 0–100.
- The number of output classes is `len(breaks) + 1` (or `len(percentiles) + 1`).
- Class values are 0-indexed unsigned 8-bit integers.
- Percentiles are computed from the *current* non-masked distribution, so they are
  inherently relative to the study area. Two different study areas will have different
  absolute thresholds even with the same percentile specification.

---

### Utility Methods

#### `raster_path(name: str) -> str`

Generate a file path for a temporary raster in the analysis working directory.

#### `verify_source(src: str)`

Validate that an external raster aligns spatially (extent, cell size, projection, shape)
with the study area DEM. Raises `ValueError` on mismatch.

#### `calc_slope()`

Derive topographic slope (degrees) from the DEM and set `self.slope`, if it has not
already been computed.

---

## Complete Example

```python
from hydrotools.morphology import RiparianConnectivity

dem = "/data/dem.tif"
precip = "/data/mean_annual_precip_mm.tif"
output = "/data/riparian_connectivity.gpkg"

# 1. Initialize with DEM (projected CRS, metres)
rc = RiparianConnectivity(dem)

# 2. Extract stream network (1 km² minimum contributing area)
rc.extract_streams(min_ws_area=1e6)

# 3. Calculate the stream attributes that define the riparian reach
rc.calc_bankfull_width(precip)
rc.calc_stream_slope()

# 4. Define riparian region — inspect rc.region and iterate if needed
rc.define_region(width_factor=3.0, vertical_weight=20.0)

# 5. Generate connectivity output
rc.calc_connectivity(output)
```

### Custom Classification

```python
# Quartile-based classification (4 classes)
rc.calc_connectivity(
    output,
    percentiles=[25, 50, 75],
)

# Continuous raster output (no classification)
rc.calc_connectivity(
    output,
    quantize=False,
    vector=False,
)
```

---

## Notes and Considerations

1. **Defaults require calibration.** `width_factor`, `slope_cutoff`, and
   `vertical_weight` are starting points, not empirically fitted values. Small streams
   with a bankfull width of 1–3 m get a reach of only a few metres with the default
   width factor, which may be a single cell on coarser DEMs. Calibrate against observed
   valley bottoms or field-mapped riparian extents.

2. **Watershed-to-watershed comparability.** Region values and the default class
   breaks are absolute, so they are comparable when the same parameters (including
   `max_reach`) and bankfull width coefficients are used. Percentile classification
   breaks this. DEM resolution and source also affect slope, and therefore both the
   cost surface and stream slope, so compare watersheds derived from consistent DEMs.
   Watersheds with only small streams will have low values throughout, which is the
   intended result of an absolute index.

3. **Stream density is no longer a criterion.** The previous multi-criteria aggregation
   (TWI, bankfull width, stream slope, and stream density) has been replaced by the
   least-cost expansion. Bankfull width and stream slope now live in the reach seeded at
   streams, and terrain slope lives in the cost surface. The standalone
   `morphology.topographic_wetness` function is still available.

4. **Temporary directory lifecycle.** The `TemporaryDirectory` is created in `__init__`
   but there is no explicit cleanup (`__del__`, `close`, or context manager protocol).
   The directory is cleaned up when the `TemporaryDirectory` object is garbage
   collected, but explicit lifecycle management (e.g., implementing `__enter__` /
   `__exit__`) would be more robust.

5. **No `save` / `load` persistence.** The commented-out `save()` and
   `load_from_data()` stubs indicate planned but unimplemented persistence. All
   intermediate rasters live in a temp directory and are lost when the object is
   destroyed.

---

## Dependencies

| Dependency   | Purpose |
|--------------|---------|
| GRASS GIS    | `r.watershed`, `r.stream.extract`, `r.cost`, `r.flowaccumulation` |
| GDAL         | `gdaldem slope`, raster I/O |
| dask         | Lazy chunked array computation |
| scipy        | `griddata` interpolation |
| scikit-learn | `BallTree` for nearest-neighbour queries |
| numba        | JIT compilation for pixel-level algorithms |
| numpy        | Array operations |

---

## References

- Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable contributing
  area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43–69.

- Church, M. (2002). Geomorphic thresholds in riverine landscapes. *Freshwater
  Biology*, 47(4), 541–557.

- Abood, S. A., Maclean, A. L., & Mason, L. A. (2012). Modeling riparian zones
  utilizing DEMs and flood height data via GIS. *Photogrammetric Engineering & Remote
  Sensing*, 78(3), 259–269.

- Adriaensen, F., Chardon, J. P., De Blust, G., Swinnen, E., Villalba, S.,
  Gulinck, H., & Matthysen, E. (2003). The application of 'least-cost' modelling as
  a functional landscape model. *Landscape and Urban Planning*, 64(4), 233–247.

- Dunne, T., & Leopold, L. B. (1978). *Water in Environmental Planning*. W. H. Freeman.

- Eastman, J. R. (1999). Multi-criteria evaluation and GIS. In P. A. Longley, M. F.
  Goodchild, D. J. Maguire, & D. W. Rhind (Eds.), *Geographical Information Systems:
  Principles, Techniques, Management and Applications* (Vol. 1, pp. 493–502). Wiley.

- Gallant, J. C., & Dowling, T. I. (2003). A multiresolution index of valley bottom
  flatness for mapping depositional areas. *Water Resources Research*, 39(12), 1347.

- Gilbert, J. T., Macfarlane, W. W., & Wheaton, J. M. (2016). The Valley Bottom
  Extraction Tool (V-BET): a GIS tool for delineating valley bottoms across entire
  drainage networks. *Computers & Geosciences*, 97, 1–14.

- Grabs, T., Seibert, J., Bishop, K., & Laudon, H. (2009). Modeling spatial patterns
  of saturated areas: a comparison of the topographic wetness index and a dynamic
  distributed model. *Journal of Hydrology*, 373(1–2), 15–23.

- Gregory, S. V., Swanson, F. J., McKee, W. A., & Cummins, K. W. (1991). An
  ecosystem perspective of riparian zones. *BioScience*, 41(8), 540–551.

- Hall, J. E., Holzer, D. M., & Beechie, T. J. (2007). Predicting river floodplain
  and lateral channel migration for salmon habitat conservation. *Journal of the
  American Water Resources Association*, 43(3), 786–797.

- Jenks, G. F. (1967). The data model concept in statistical mapping. *International
  Yearbook of Cartography*, 7, 186–190.

- Jencso, K. G., McGlynn, B. L., Gooseff, M. N., Wondzell, S. M., Bencala, K. E.,
  & Marshall, L. A. (2009). Hydrologic connectivity between landscapes and streams:
  transferring reach- and plot-scale understanding to the catchment scale. *Water
  Resources Research*, 45(4), W04428.

- Kopecký, M., & Čížková, Š. (2010). Using topographic wetness index in vegetation
  ecology: does the algorithm matter? *Applied Vegetation Science*, 13(4), 450–459.

- Leopold, L. B., & Maddock, T. (1953). The hydraulic geometry of stream channels and
  some physiographic implications. *USGS Professional Paper 252*.

- McRae, B. H., Dickson, B. G., Keitt, T. H., & Shah, V. B. (2008). Using circuit
  theory to model connectivity in ecology, evolution, and conservation. *Ecology*,
  89(10), 2712–2724.

- Montgomery, D. R., & Buffington, J. M. (1997). Channel-reach morphology in mountain
  drainage basins. *Geological Society of America Bulletin*, 109(5), 596–611.

- Naiman, R. J., & Décamps, H. (1997). The ecology of interfaces: riparian zones.
  *Annual Review of Ecology and Systematics*, 28, 621–658.

- Naiman, R. J., Décamps, H., & McClain, M. E. (2005). *Riparia: Ecology,
  Conservation, and Management of Streamside Communities*. Elsevier Academic Press.

- O'Callaghan, J. F., & Mark, D. M. (1984). The extraction of drainage networks from
  digital elevation data. *Computer Vision, Graphics, and Image Processing*, 28(3),
  323–344.

- Sørensen, R., Zinko, U., & Seibert, J. (2006). On the calculation of the
  topographic wetness index: evaluation of different methods based on field
  observations. *Hydrology and Earth System Sciences*, 10(1), 101–112.

- Tarboton, D. G., Bras, R. L., & Rodriguez-Iturbe, I. (1991). On the extraction of
  channel networks from digital elevation data. *Hydrological Processes*, 5(1), 81–100.

- Tockner, K., & Stanford, J. A. (2002). Riverine flood plains: present state and
  future trends. *Environmental Conservation*, 29(3), 308–330.
