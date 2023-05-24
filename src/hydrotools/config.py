from tempfile import gettempdir


# Reading raster data using dask loads the .tif header information with every
# call for data. As such, use larger chunks to achieve better performance.
CHUNKS = (1, 4096, 4096)

TMP_DIR = gettempdir()

GRASS_TMP: str = gettempdir()

GRASS_FLAGS: dict = {"quiet": True}

GDALWARP_ARGS = [
    "-of",
    "GTiff",
    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
    "-co",
    "BLOCKXSIZE=512",
    "-co",
    "BLOCKYSIZE=512",
    "-co",
    "BIGTIFF=YES",
    "-q",
]

COG_ARGS = [
    "-of",
    "COG",
    "-co",
    "BIGTIFF=YES",
    "-q",
]
