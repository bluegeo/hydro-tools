# Riparian Connectivity Analysis

## Overview

The `RiparianConnectivity` class in `hydrotools.morphology` provides an end-to-end
workflow for delineating and classifying riparian zones based on multi-attribute
sensitivity analysis. Starting from a Digital Elevation Model (DEM), the workflow
derives hydrological, topographic, and geomorphic attributes, normalizes them to a
common scale, spatially interpolates them across a defined riparian extent, and combines
them into a composite connectivity index that can be exported as a raster or vector
dataset.

The approach is grounded in the principle that riparian connectivity — the degree of
lateral hydrological and ecological linkage between a stream and its floodplain — can
be characterized by combining proxy indicators of channel geometry, topographic
position, and drainage density (Naiman & Décamps, 1997; Gregory et al., 1991).

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

### Topographic Wetness Index (TWI)

The TWI variant used here is a **cost-distance-based** index, not the classical
Beven & Kirkby (1979) formulation ($\ln(a / \tan\beta)$). Instead, it uses a Least
Cost Path (LCP) surface computed from streams using slope as a friction surface via
GRASS GIS `r.cost`. The accumulated cost is then inverted so that low-slope, near-stream
areas receive the highest values. This produces a surface that conceptually captures
the same topographic convergence and wetness tendency as the classical TWI, but is more
sensitive to the lateral connectivity between streams and valley bottoms.

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

#### Implementation details

The GRASS `r.cost` algorithm uses Dijkstra's shortest path to compute cumulative cost
from source cells (stream locations). The knight's move option extends the 8-connected
neighbourhood to include cells reachable by a chess knight's L-shaped move, improving
angular resolution of the cost surface and reducing directional bias inherent in
raster-based distance computations (GRASS GIS Reference Manual).

The cost surface is constructed as:

$$
C_{ij} = \frac{\text{slope}_{ij}}{90°} \times \bar{s}
$$

Where $\text{slope}_{ij}$ is the topographic slope in degrees at cell $(i,j)$ and
$\bar{s}$ is the average cell size, converting the dimensionless normalized slope
into a distance-scaled friction. Stream cells are assigned zero cost (they are the
source). The cumulative cost surface is then inverted:

$$
\text{TWI}_{ij} = C_{\max} - C_{ij} \quad \text{where } C_{ij} \leq C_{\max}
$$

Cells exceeding $C_{\max}$ (default: 750) are masked. The resulting surface ranges
from 0 (at the cost threshold boundary) to $C_{\max}$ (at stream locations), with
higher values indicating greater topographic connectivity to the channel network.

### Stream Slope

Stream slope is derived by masking the DEM to only stream-cell locations and computing
slope (via GDAL `gdaldem slope`) on the masked surface. This yields local longitudinal
gradient at each stream pixel. An optional focal-mean smoothing step averages slope
values over a specified neighbourhood, producing segment-representative gradients rather
than cell-level noise.

In the connectivity workflow, stream slope is **inverted** during normalization: steeper
streams receive lower connectivity scores, reflecting the principle that low-gradient
streams are more closely associated with broad, well-connected floodplains (Montgomery
& Buffington, 1997).

### Stream Density

Stream density is computed as a focal sum of stream-cell presence within a circular
kernel (default radius: 500 m), normalized by the kernel area. This is functionally
equivalent to a moving-window drainage density calculation:

$$
D_d = \frac{\sum \text{stream cells within radius}}{\text{total cells within radius}}
$$

Higher stream density indicates a more dissected landscape with greater potential for
lateral connectivity.

### Normalization and Interpolation

Before attributes can be combined, they must be brought to a common, dimensionless
scale. This step is essential in any multi-criteria analysis (MCA) to prevent
attributes with larger numerical ranges from dominating the composite score (Malczewski,
1999; Voogd, 1983).

The normalization applied here is a **linear score transformation** (also called
*value function* or *benefit/cost scaling*) that maps each raw attribute into the
range [0, 1]:

$$
v_i(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

Where $x_{\min}$ and $x_{\max}$ are user-specified cutoff values that define the
full range of each criterion. Values below $x_{\min}$ map to 0 and values above
$x_{\max}$ map to 1. This is a standard approach in spatial MCA literature
(Eastman, 1999; Malczewski, 2000).

For **cost criteria** — attributes where lower values indicate higher connectivity
(e.g. stream slope) — the transformation is inverted:

$$
v_i(x) = 1 - \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

This ensures that all normalized layers follow the same directionality: higher values
always indicate a greater contribution to connectivity.

Once normalized, stream-cell values are **spatially interpolated** across the full
riparian region using Inverse Distance Weighting (IDW). The IDW interpolant uses a
neighbourhood radius of approximately 200 m (automatically scaled by cell size) to
spread channel-derived attributes into the surrounding floodplain. This step converts
point-like stream attributes into continuous surfaces that cover the riparian extent,
allowing pixel-wise combination.

### Composite Index: Multi-Criteria Aggregation

The final connectivity index is produced using the **Simple Additive Weighting (SAW)**
method — also known as the *Weighted Linear Combination (WLC)* — which is the most
widely used aggregation rule in spatial multi-criteria decision analysis (Malczewski,
2000; Jiang & Eastman, 2000).

The general SAW/WLC formulation is:

$$
S = \sum_{i=1}^{n} w_i \cdot v_i
$$

Where:
- $S$ is the composite suitability (connectivity) score
- $w_i$ is the weight of the $i$-th criterion
- $v_i$ is the normalized score of the $i$-th criterion at each cell
- $n$ is the number of criteria

In the current implementation, **all weights are equal** ($w_i = 1$ for all $i$),
reducing the equation to a simple unweighted sum. This is equivalent to assuming that
each attribute contributes equally to riparian connectivity — a reasonable default when
no empirical basis exists for preferring one criterion over another (Jiang & Eastman,
2000).

The four default criteria and their roles in the additive model are:

| Criterion | Type | Rationale |
|-----------|------|-----------|
| TWI (region) | Benefit | Topographic proximity and slope-weighted distance to streams; captures valley-bottom position. Higher values = closer to streams on flatter terrain. |
| Bankfull width | Benefit | Channel size as a proxy for floodplain extent and lateral hydrologic exchange. Larger channels = broader active floodplains (Leopold & Maddock, 1953; Hall et al., 2007). |
| Stream slope (inverted) | Benefit (after inversion) | Longitudinal gradient as a proxy for channel-floodplain coupling. Low-gradient reaches are associated with wider valleys and greater lateral connectivity (Montgomery & Buffington, 1997). |
| Stream density | Benefit | Local drainage density reflecting landscape dissection and hydrologic network complexity. Higher density suggests more pathways for lateral exchange. |

**Why SAW?** SAW satisfies the axioms of *compensatory* aggregation — a low score on one
criterion can be offset by a high score on another (Yager, 1988). This is appropriate
for riparian connectivity because the underlying processes are partially substitutable:
a narrow channel on very flat terrain may exhibit similar connectivity to a wider channel
on moderate terrain. Non-compensatory methods (e.g. OWA with high `orness`) would be
more appropriate if connectivity required *all* criteria to score well simultaneously.

### Percentile-Based Classification

The continuous composite surface is optionally discretized into ordinal zones using
percentile thresholds computed from the distribution of non-masked values. The default
tercile boundaries ($P_{33.3}$ and $P_{66.7}$) divide the surface into three classes:

| Class | Percentile range | Interpretation |
|-------|-----------------|----------------|
| 0     | $< P_{33.3}$   | Low connectivity |
| 1     | $P_{33.3}$ – $P_{66.7}$ | Moderate connectivity |
| 2     | $> P_{66.7}$    | High connectivity |

Percentile-based thresholds are distribution-relative rather than absolute, meaning
each class will contain approximately one-third of the riparian area by default. This
approach is robust to study-area differences in attribute magnitude and is recommended
when absolute thresholds are unknown (Jenks, 1967; Eastman, 1999). Users can supply
custom percentile lists to create any number of classes (e.g. quartiles, quintiles).

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
 │       └── flow accumulation raster
 │
 ├─► calc_twi()
 │       │
 │       ├── slope raster (derived internally)
 │       └── TWI raster (cost-distance from streams, inverted)
 │
 ├─► define_region(twi_cutoff)
 │       │
 │       └── normalized TWI region mask (riparian extent)
 │
 ├─► calc_bankfull_width(annual_precip)
 │       │
 │       └── bankfull width → normalized → interpolated into region
 │
 ├─► calc_stream_slope()
 │       │
 │       └── stream slope → normalized (inverted) → interpolated into region
 │
 ├─► calc_stream_density()
 │       │
 │       └── stream density → normalized → interpolated into region
 │
 └─► calc_connectivity(output_path)
         │
         └── sum of attributes → percentile classification → vector/raster
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
- Initializes all attribute slots (`slope`, `flow_accumulation`, `streams`, `twi`,
  `region`, `bankfull_width`, `stream_slope`, `stream_density`) to `None`.

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

### Step 2: Topographic Wetness Index

#### `calc_twi(memory=4096)`

Compute a cost-distance-based Topographic Wetness Index from the stream network.

**Parameters:**
| Parameter | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `memory`  | `int` | `4096`  | Memory allocation (MB) for GRASS `r.cost`. |

**Behaviour:**
1. If slope has not been computed, derives it from the DEM using `elevation.slope()`.
2. Constructs a cost surface: slope (degrees) is normalized to [0, 1] and scaled by
   the average cell size.
3. Sets cost to 0 at stream locations (streams are the origin of the LCP surface).
4. Runs GRASS `r.cost` with the knight's move flag to compute cumulative cost from
   streams across the landscape.
5. Inverts the cost surface: `TWI = max_cost - cost`, masking cells that exceed a
   maximum cost threshold (default: 750).
6. Sets `self.twi` and `self.slope`.

**Prerequisite:** `extract_streams()` must have been called.

---

### Step 3: Riparian Region Definition

#### `define_region(twi_cutoff=745.0)`

Threshold and normalize the TWI to delineate the spatial extent of riparian zones.

**Parameters:**
| Parameter    | Type    | Default | Description |
|--------------|---------|---------|-------------|
| `twi_cutoff` | `float` | `745.0` | Minimum TWI value to include in the riparian region. |

**Behaviour:**
1. Masks the TWI raster below `twi_cutoff`.
2. Normalizes remaining values to [0, 1] using min-max scaling
   ($\frac{TWI - cutoff}{TWI_{max} - cutoff}$).
3. Saves the result as the region raster and sets `self.region`.

**Iterative calibration:** This method is designed to be called repeatedly. After each
call, inspect `self.region` in a GIS viewer to verify the riparian extent aligns with
observed valley bottoms. Adjust `twi_cutoff` and re-run as needed.

**Prerequisite:** `calc_twi()` must have been called.

---

### Step 4a: Bankfull Width (optional attribute)

#### `calc_bankfull_width(annual_precip_src: str, cutoff=10.0)`

Compute empirical bankfull width along streams, normalize, and interpolate across the
riparian region.

**Parameters:**
| Parameter           | Type    | Default | Description |
|---------------------|---------|---------|-------------|
| `annual_precip_src` | `str`   | —       | Path to a mean annual precipitation raster in mm. Must align spatially with the DEM. |
| `cutoff`            | `float` | `10.0`  | Maximum bankfull width (m) for normalization. Values above this are clamped to 1.0. |

**Behaviour:**
1. Validates spatial alignment of the precipitation raster with the study area.
2. Computes contributing area (km²) and mean annual precipitation (cm/yr) along streams
   using flow accumulation routing (`FlowAccumulation.contributing_area` and
   `FlowAccumulation.calculate` with `method="mean"`).
3. Applies the Hall et al. (2007) equation:
   $W_{bf} = 0.042 \cdot A^{0.48} \cdot P^{0.74}$
4. Normalizes to [0, 1] with the specified cutoff.
5. Interpolates normalized values from stream cells into the full riparian region using
   IDW (`PointInterpolator.idw`).
6. Sets `self.bankfull_width`.

**Prerequisites:** `extract_streams()` and `define_region()`.

**Reference:**
> Hall, J. E., Holzer, D. M., & Beechie, T. J. (2007). Predicting river floodplain
> and lateral channel migration for salmon habitat conservation. *JAWRA*, 43(3),
> 786–797.

---

### Step 4b: Stream Slope (optional attribute)

#### `calc_stream_slope(cutoff=15.0)`

Compute longitudinal slope along stream cells, normalize (inverted), and interpolate
across the riparian region.

**Parameters:**
| Parameter | Type    | Default | Description |
|-----------|---------|---------|-------------|
| `cutoff`  | `float` | `15.0`  | Maximum slope in degrees for normalization. |

**Behaviour:**
1. Masks the DEM to stream-cell locations only.
2. Computes slope on the masked DEM using GDAL `gdaldem slope`.
3. Normalizes to [0, 1] with the specified cutoff, **inverted** so that low-slope
   stream reaches score higher (flatter streams → broader floodplains → higher
   connectivity).
4. Interpolates into the riparian region via IDW.
5. Sets `self.stream_slope`.

**Prerequisites:** `extract_streams()` and `define_region()`.

**Scientific note:** The inverse relationship between channel gradient and floodplain
width is a fundamental tenet of fluvial geomorphology (Montgomery & Buffington, 1997;
Church, 2002).

---

### Step 4c: Stream Density (optional attribute)

#### `calc_stream_density(sample_distance=500, cutoff=2e5)`

Compute local drainage density, normalize, and interpolate across the riparian region.

**Parameters:**
| Parameter         | Type    | Default | Description |
|-------------------|---------|---------|-------------|
| `sample_distance` | `float` | `500`   | Radius (m) of the circular focal kernel for counting stream cells. |
| `cutoff`          | `float` | `2e5`   | Maximum stream density value for normalization. This is expressed in cells assuming 1 m cell size; internally adjusted for actual cell size. |

**Behaviour:**
1. Creates a boolean stream mask (True where streams exist).
2. Builds a circular kernel from `sample_distance` and the raster cell size.
3. Applies a focal sum filter (convolution) to count stream cells within the kernel at
   each location.
4. Divides by kernel area to get a density proportion.
5. Adjusts the cutoff for actual cell size: `cutoff / avg_cs²`.
6. Normalizes to [0, 1] with the adjusted cutoff.
7. Interpolates into the riparian region via IDW.
8. Sets `self.stream_density`.

**Prerequisites:** `extract_streams()` and `define_region()`.

---

### Step 5: Connectivity Calculation

#### `calc_connectivity(connectivity_dst, attributes=["twi", "bankfull_width", "stream_slope", "stream_density"], quantize=True, percentiles=[(1/3)*100, (2/3)*100], vector=True)`

Combine normalized attribute layers into a composite riparian connectivity index.

**Parameters:**
| Parameter          | Type   | Default | Description |
|--------------------|--------|---------|-------------|
| `connectivity_dst` | `str`  | —       | Output file path. Vector (GeoPackage) if `vector=True`, raster (GeoTIFF) otherwise. |
| `attributes`       | `list` | `["twi", "bankfull_width", "stream_slope", "stream_density"]` | Names of attributes to include in the composite index. |
| `quantize`         | `bool` | `True`  | Whether to classify the continuous index into discrete zones using percentile thresholds. |
| `percentiles`      | `list` | `[33.3, 66.7]` | Percentile boundaries for classification. Default terciles produce 3 classes. |
| `vector`           | `bool` | `True`  | Export as vector (requires `quantize=True`). |

**Behaviour:**
1. Validates that all requested attributes have been computed.
2. Resolves attribute names to raster paths. The special name `"twi"` maps to
   `self.region` — the normalized TWI surface produced by `define_region()`. All
   other names map directly (e.g. `"bankfull_width"` → `self.bankfull_width`).
3. Loads each attribute raster as a dask masked array and performs **element-wise
   summation** (Simple Additive Weighting with equal unit weights). The resulting
   continuous surface $S$ satisfies:
   $$S = \sum_{i=1}^{n} v_i$$
   where each $v_i \in [0, 1]$ is the normalized, interpolated attribute layer.
   The theoretical range of $S$ is $[0, n]$ where $n$ is the number of attributes.
4. If `quantize=True`:
   a. Extracts all non-masked values from $S$.
   b. Computes percentile thresholds $q_1, q_2, \ldots$ from the provided
      `percentiles` list using `dask.array.percentile`.
   c. Applies `np.digitize(S, [q_1, q_2, ...])` to assign each cell to an ordinal
      class. For the default terciles, this produces classes 0 (low), 1 (moderate),
      and 2 (high).
   d. Re-masks the classified surface to the riparian region extent.
5. Output:
   - If `vector=True` (requires `quantize=True`): writes the classified raster to a
     temporary file, then vectorizes with smoothed corners to produce a polygon
     GeoPackage.
   - If `vector=False`: writes the result (continuous or classified) directly as a
     Cloud-Optimized GeoTIFF.

**Prerequisites:** All attributes listed in the `attributes` parameter must have been
computed via their respective `calc_*` methods.

**Multi-criteria method:** This is a Simple Additive Weighting (SAW) / Weighted Linear
Combination (WLC) approach with equal weights (Malczewski, 2000). See the
[Composite Index](#composite-index-multi-criteria-aggregation) section for theoretical
detail.

**Sensitivity to attribute selection:** Because of equal weighting, the relative
influence of each criterion is $1/n$. Adding or removing an attribute changes the
weight of all others. For example, using all four default attributes gives each a 25%
weight; using only TWI and bankfull width gives each 50%. Users should consider whether
this implicit rebalancing is appropriate for their application.

**Percentile classification notes:**
- The `percentiles` list must be monotonically increasing and all values ≤ 100.
- The number of output classes is `len(percentiles) + 1`.
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

#### `interpolate_into_region(src, dst)`

Spatially interpolate values from a source raster (typically stream-cell attributes)
to all cells in the riparian region using IDW interpolation. The number of neighbours
is automatically scaled based on cell size (~200 m radius).

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

# 3. Compute Topographic Wetness Index
rc.calc_twi()

# 4. Define riparian region — inspect rc.region and iterate if needed
rc.define_region(twi_cutoff=745.0)

# 5. Calculate optional attributes
rc.calc_bankfull_width(precip)
rc.calc_stream_slope()
rc.calc_stream_density()

# 6. Generate connectivity output
rc.calc_connectivity(output)
```

### Minimal Example (TWI-only connectivity)

```python
rc = RiparianConnectivity(dem)
rc.extract_streams()
rc.calc_twi()
rc.define_region()
rc.calc_connectivity(output, attributes=["twi"], vector=False)
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

## Logic Assessment and Notes

### Strengths

1. **Well-structured stateful workflow.** The class enforces a logical ordering of
   operations via prerequisite checks (`AttributeError` guards). Each method validates
   that required prior steps have been completed.

2. **Iterative region calibration.** `define_region()` is designed for repeated
   invocation with different thresholds, supporting visual verification — a practical
   approach for a parameter that is inherently site-specific.

3. **Normalization before combination.** All attributes are normalized to [0, 1] before
   summation, ensuring equal weighting regardless of original units or magnitude. The
   inversion of stream slope correctly reflects its inverse relationship with
   connectivity.

4. **Lazy evaluation.** Dask arrays are used throughout, enabling processing of
   rasters larger than available memory via chunked computation.

5. **Spatial alignment validation.** `verify_source()` checks extent, cell size,
   projection, and shape before accepting external data, preventing misaligned inputs.

### Observations and Considerations

1. **Attribute weighting is uniform.** The additive combination in `calc_connectivity`
   uses Simple Additive Weighting (SAW) with equal unit weights ($w_i = 1$). This is
   the simplest and most transparent form of multi-criteria aggregation, but it assumes
   all criteria are equally important. If domain knowledge or empirical evidence
   suggests otherwise, consider extending the method to accept user-specified weights.
   Techniques such as the Analytic Hierarchy Process (AHP; Saaty, 1980) or rank-order
   weighting (Malczewski, 1999) could provide a principled basis for unequal weights.

2. **TWI is both a region mask and a connectivity attribute.** When `"twi"` is in the
   attribute list, `calc_connectivity` uses `self.region` (the normalized TWI) as both
   the spatial mask and one of the summed layers. This is a deliberate design choice
   that "double-counts" TWI: once as the region extent and once as a continuous
   attribute. This behaviour is intentional — areas near streams with flat terrain
   inherently score higher — but users should be aware of this coupling.

3. **`interpolate_into_region` opens the source raster twice.** Two `Raster(src)`
   instances are created on consecutive lines:
   ```python
   obs_raster = Raster(src)
   obs, obs_points = Raster(src).data_and_index()
   ```
   This opens and reads data from the same file twice. The first instance is only used
   for its `shape` attribute later.

4. **`define_region` default vs. docstring mismatch.** The function signature defaults
   `twi_cutoff` to `745.0`, but the docstring says "Defaults to 830." The signature
   value takes precedence at runtime.

5. **Temporary directory lifecycle.** The `TemporaryDirectory` is created in `__init__`
   but there is no explicit cleanup (`__del__`, `close`, or context manager protocol).
   The directory is cleaned up when the `TemporaryDirectory` object is garbage
   collected, but explicit lifecycle management (e.g., implementing `__enter__` /
   `__exit__`) would be more robust.

6. **No `save` / `load` persistence.** The commented-out `save()` and
   `load_from_data()` stubs indicate planned but unimplemented persistence. All
   intermediate rasters live in a temp directory and are lost when the object is
   destroyed.

7. **`extract_streams` produces temporary flow direction grids.** Two temporary flow
   direction rasters (`fd1`, `fd2`) are created and discarded. Only the final
   `streams` and `flow_accumulation` paths are retained. The flow direction is needed
   by `calc_bankfull_width` (via `FlowAccumulation`), and this works because
   `self.flow_accumulation` is set — but the *direction* raster (`fd2`) is not stored.
   The `bankfull_width` function internally uses `FlowAccumulation(flow_direction)`,
   which expects a flow *direction* raster, not an accumulation raster. However,
   `calc_bankfull_width` passes `self.flow_accumulation` as the `flow_direction`
   argument to the module-level `bankfull_width()` function. This relies on the fact
   that `FlowAccumulation.__init__` derives its needed inputs from whatever direction
   source is provided; verify that `self.flow_accumulation` is a valid input here.

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

- Jiang, H., & Eastman, J. R. (2000). Application of fuzzy measures in multi-criteria
  evaluation in GIS. *International Journal of Geographical Information Science*,
  14(2), 173–184.

- Kopecký, M., & Čížková, Š. (2010). Using topographic wetness index in vegetation
  ecology: does the algorithm matter? *Applied Vegetation Science*, 13(4), 450–459.

- Leopold, L. B., & Maddock, T. (1953). The hydraulic geometry of stream channels and
  some physiographic implications. *USGS Professional Paper 252*.

- Malczewski, J. (1999). *GIS and Multicriteria Decision Analysis*. John Wiley & Sons.

- Malczewski, J. (2000). On the use of weighted linear combination method in GIS:
  Common and best practice approaches. *Transactions in GIS*, 4(1), 5–22.

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

- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.

- Sørensen, R., Zinko, U., & Seibert, J. (2006). On the calculation of the
  topographic wetness index: evaluation of different methods based on field
  observations. *Hydrology and Earth System Sciences*, 10(1), 101–112.

- Tarboton, D. G., Bras, R. L., & Rodriguez-Iturbe, I. (1991). On the extraction of
  channel networks from digital elevation data. *Hydrological Processes*, 5(1), 81–100.

- Tockner, K., & Stanford, J. A. (2002). Riverine flood plains: present state and
  future trends. *Environmental Conservation*, 29(3), 308–330.

- Voogd, H. (1983). *Multicriteria Evaluation for Urban and Regional Planning*. Pion.

- Yager, R. R. (1988). On ordered weighted averaging aggregation operators in
  multicriteria decision making. *IEEE Transactions on Systems, Man, and Cybernetics*,
  18(1), 183–190.
