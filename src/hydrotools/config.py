from tempfile import gettempdir


# Reading raster data using dask loads the .tif header information with every
# call for data. As such, use larger chunks to achieve better performance.
CHUNKS = (1, 4096, 4096)

TMP_DIR = gettempdir()

GRASS_LOCATION: str = "hydro-tools-grass"

GRASS_TMP: str = gettempdir()

GDAL_DEFAULT_ARGS = [
    "-of",
    "GTiff",
    "-co",
    "TILED=YES",
    "-co",
    "COMPRESS=LZW",
    "-co",
    "BLOCKXSIZE=512",
    "-co",
    "BLOCKYSIZE=512",
    "-co",
    "BIGTIFF=YES",
    "-q",
    "-multi",
    "-wo",
    "NUM_THREADS=ALL_CPUS",
]