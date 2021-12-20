# Hydro-Tools

A collection of tools for terrain and hydrological analysis, packaged into an
opinionated python module that simplifies tasks by combining functionality from several
libraries into single abstractions. Libraries include:

- GDAL
- GRASS GIS
- rasterio
- scipy
- dask

## Usage notes

Most raster operations allow any GDAL-supported format as an input, but the format of
ouput datasets is limited to Cloud-Optimized GeoTiffs.

## Examples

### _Split a DEM into watersheds with areas above 1 km<sup>2</sup>_

```python
from hydrotools.watershed import auto_basin

# A Digital Elevation Model in a project coordinate system (m)
dem = "/data/dem.tif"
output_basins = "/data/basins.tif"

auto_basin(dem, 1E6, output_basins)
```

What if the DEM is in a GCS?
We can warp it into a temporary dataset in m.

**Note** Many hydrologic analyses use continuous data, such as a DEM.
As such, the default resampling method is a cubic spline.

```python
from hydrotools.raster import TempRasterFile, warp

with TempRasterFile() as warped_dem:
    warp(dem, warped_dem, t_srs=3857)
    auto_basin(warped_dem, 1E6, output_basins)
```

### _Simulate streams (with areas > 1km<sup>2</sup>) using a simple reclassification_

```python
import dask.array as da

from hydrotools.watershed import flow_direction_accumulation
from hydrotools.raster import Raster, from_raster, to_raster

dem = "/data/dem.tif"
output_flow_direction = "/data/flow_direction.tif"
output_flow_accumulation = "/data/flow_accumulation.tif"

# Calculate flow direction and flow accumulation, which uses
# GRASS r.watershed under the hood
flow_direction_accumulation(dem, output_flow_direction, output_flow_accumulation)

# Mask flow accumulation with values below the threshold
output_streams = "/data/streams_1km2.tif"

# Collect the raster cell size to use for a flow accumulation threshold
# Underlying raster files are not persistently open like GDAL or rasterio
accumulation_raster = Raster(output_flow_accumulation)
threshold = 1E6 / (accumulation_raster.csx * accumulation_raster.csy)

# This is a dask-like operation, which behaves like `from_array`, but
# uses a raster instead of an array. All operations are done lazily
# and no data are read from the disk until `compute`, or `to_raster`
# is called.
a = from_raster(output_flow_accumulation)

# Raster no data values are handled using dask (numpy) masked array
# logic. To add additional no data, we use the `ma` module to add to
# the mask where the flow accumulation is below the threshold.
a = da.ma.masked_where(a < threshold, a)

# To execute the dask compute graph and store the results into the
# output raster (in chunks for memory efficiency) we call `to_raster`,
# which includes a raster to use as a template for the output.
to_raster(a, output_flow_accumulation, output_streams)
```

## Configuration

Most constants referenced by the library are hosted in the `config` module.

Defaults include:

| constant          | Description                                        | Default                    |
| ----------------- | -------------------------------------------------- | -------------------------- |
| CHUNKS            | Defaults chunks for Dask                           | (1, 4096, 4096)            |
| TMP_DIR           | Temporary directory                                | `os.gettempdir()`          |
| GRASS_LOCATION    | Name of GRASS mapset on disk                       | "hydro-tools-grass"        |
| GRASS_TMP         | Temporary directory for GRASS                      | `os.gettempdir()`          |
| GDAL_DEFAULT_ARGS | Default arguments for all GDAL subprocess commands | see `hydrotools.config.py` |

## Lower-level operations

This library also includes simplified ways of abstracting tools like GRASS, GDAL, and
dask. Below are some examples that utilize these resources.

#### Preparing rasters

_Docs in progress..._

#### Raster math using dask

_Docs in progress..._

#### Rasters and no data

_Docs in progress..._

#### Running GRASS commands

_Docs in progress..._
