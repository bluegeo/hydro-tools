# Hydro-Tools

Hydrological and terrain utilities for preparing data and performing simulations. Hydro-Tools combines GDAL, GRASS GIS, rasterio, scipy, dask, and numba into a single opinionated Python library that simplifies complex geospatial workflows into concise function calls.

## Key Features

- **Watershed Analysis** — Flow direction & accumulation, stream extraction, basin delineation, and stream ordering (Strahler/Horton) via GRASS GIS
- **Terrain Derivatives** — Slope, aspect, solar radiation, Terrain Ruggedness Index (TRI), and DEM conditioning
- **Stream Network Characterization** — Topology tracing, reach vectorization, sinuosity, bankfull width/depth estimation, valley confinement, and multi-attribute annotation
- **Riparian Connectivity** — End-to-end workflow for riparian zone delineation using Topographic Wetness Index, bankfull dimensions, stream density, and multi-criteria aggregation (Simple Additive Weighting)
- **Spatial Interpolation** — Inverse Distance Weighting (IDW), cubic splines, moving-window statistical filters, distance transforms, and nodata filling
- **Raster I/O & Math** — Lazy dask-backed masked arrays with chunked reading/writing, automatic nodata propagation, warping, clipping, alignment, and vectorization
- **Cloud-Optimized Output** — All raster outputs default to Cloud-Optimized GeoTIFF (COG); all vector outputs default to GeoPackage
- **GRASS GIS Integration** — Managed GRASS sessions via `GrassRunner` context manager with automatic setup and cleanup
- **LiDAR Processing** — Generate Digital Terrain Models from .las point clouds using GRASS edge detection and surface interpolation

## Benefits

- **Simplified API** — Complex multi-step geospatial operations reduced to single function calls
- **Memory-Efficient** — Dask lazy evaluation and chunked I/O handle rasters larger than available RAM
- **Reproducible Pipelines** — Consistent configuration, automatic temporary file cleanup, and deterministic outputs
- **JIT-Compiled Performance** — Numba-accelerated inner loops for topology tracing, flow routing, and custom raster filters
- **Interoperable** — Accepts any GDAL/OGR-supported format as input; outputs industry-standard COG and GeoPackage formats
- **Docker-Ready** — Includes a Dockerfile with all system dependencies (GDAL, GRASS GIS) pre-configured

## Installation

### Prerequisites

1. **GDAL** with Python bindings (conda is recommended)
2. **GRASS GIS** — [download here](https://grass.osgeo.org/download/)

### Install from source

```bash
pip install .
```

### Docker

```bash
docker build -t hydro-tools .
docker run -it hydro-tools
```

The container starts an interactive Python shell with `hydrotools` and all dependencies pre-installed.

## Modules

| Module | Description |
| --- | --- |
| `hydrotools.watershed` | Flow routing, stream extraction, basin delineation, flow accumulation statistics |
| `hydrotools.elevation` | Slope, aspect, solar radiation, TRI, DEM alignment, LiDAR-to-DTM |
| `hydrotools.morphology` | Bankfull width/depth, valley confinement, topographic wetness, riparian connectivity |
| `hydrotools.streams` | Stream topology, Strahler ordering, reach vectorization, comprehensive watershed attribute extraction |
| `hydrotools.interpolate` | IDW, cubic splines, moving-window filters, distance transforms, nodata filling, normalization |
| `hydrotools.raster` | Raster I/O (`from_raster`/`to_raster`), warping, clipping, alignment, vectorization, `Raster` class |
| `hydrotools.utils` | GRASS session management, temporary file context managers, coordinate transforms, kernel generation |
| `hydrotools.config` | Centralized constants for dask chunks, temp directories, GRASS flags, GDAL/COG arguments |

## Quick Start

### Delineate watersheds above 1 km²

```python
from hydrotools.watershed import auto_basin

auto_basin("/data/dem.tif", 1E6, "/data/basins.tif")
```

### Extract a stream network and compute flow accumulation

```python
from hydrotools.watershed import flow_direction_accumulation, extract_streams

flow_direction_accumulation("/data/dem.tif", "/data/fd.tif", "/data/fa.tif")
extract_streams("/data/dem.tif", "/data/fa.tif", "/data/streams.tif", "/data/fd.tif", min_area=1E6)
```

### Lazy raster math with dask

```python
import dask.array as da
from hydrotools.raster import Raster, from_raster, to_raster

# Load raster lazily — no data read until compute
a = from_raster("/data/flow_accumulation.tif")

# Mask cells below a contributing area threshold
r = Raster("/data/flow_accumulation.tif")
threshold = 1E6 / (r.csx * r.csy)
a = da.ma.masked_where(a < threshold, a)

# Write result as Cloud-Optimized GeoTIFF
to_raster(a, "/data/flow_accumulation.tif", "/data/streams.tif")
```

### Warp a raster to a different projection

```python
from hydrotools.raster import warp
from hydrotools.utils import TempRasterFile

with TempRasterFile() as warped:
    warp("/data/dem.tif", warped, t_srs=3857)
    # Use warped raster for further analysis...
```

### Calculate terrain derivatives

```python
from hydrotools.elevation import slope, aspect, solar_radiation, terrain_ruggedness_index

slope("/data/dem.tif", "/data/slope.tif")
aspect("/data/dem.tif", "/data/aspect.tif")
terrain_ruggedness_index("/data/dem.tif", "/data/tri.tif")

# Daily solar radiation for day 172 (summer solstice) with 0.5-hour steps
solar_radiation("/data/dem.tif", day=172, step=0.5, rad_dst="/data/solar.tif")
```

### Riparian connectivity analysis

```python
from hydrotools.morphology import RiparianConnectivity

rc = RiparianConnectivity("/data/dem.tif")
rc.extract_streams(min_ws_area=1E6)
rc.calc_twi()
rc.define_region(twi_cutoff=745.0)
rc.calc_bankfull_width("/data/annual_precip.tif")
rc.calc_stream_slope()
rc.calc_stream_density(sample_distance=500)
rc.calc_connectivity("/data/connectivity.tif")
```

### Run a GRASS command directly

```python
from hydrotools.utils import GrassRunner

with GrassRunner("/data/dem.tif") as gs:
    gs.run_command("r.neighbors", input="dem", output="smoothed", size=5, method="average")
    gs.save_raster("smoothed", "/data/smoothed.tif")
```

## Configuration

Runtime constants are centralized in `hydrotools.config` and can be modified at import time:

| Constant | Description | Default |
| --- | --- | --- |
| `CHUNKS` | Dask chunk size for raster operations | `(1, 4096, 4096)` |
| `TMP_DIR` | Temporary directory for intermediate files | `os.gettempdir()` or `$HYDROTOOLS_TMP_DIR` |
| `GRASS_TMP` | Temporary directory for GRASS sessions | `os.gettempdir()` |
| `GRASS_FLAGS` | Default GRASS command flags | `{"quiet": True}` |
| `GDALWARP_ARGS` | GDAL warp arguments (LZW, tiled 512×512, BigTIFF) | See `hydrotools.config` |
| `COG_ARGS` | Cloud-Optimized GeoTIFF output arguments | See `hydrotools.config` |

## Dependencies

- **rasterio** — Raster I/O
- **pyproj** — Projection handling
- **grass-session** — GRASS GIS integration
- **dask** / **dask-image** — Lazy parallel computation and morphological operations
- **numpy** (<2.2) — Array operations
- **numba** — JIT-compiled inner loops
- **scikit-learn** — KNN and BallTree for spatial interpolation
- **scikit-image** — Image processing utilities
- **fiona** — Vector I/O
- **shapely** / **rtree** — Geometry operations and spatial indexing
- **matplotlib** — Plotting

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
