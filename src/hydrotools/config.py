import os
from tempfile import gettempdir


# Reading raster data using dask loads the .tif header information with every
# call for data. As such, use larger chunks to achieve better performance.
CHUNKS = (1, 4096, 4096)

TMP_DIR: str = (
    os.environ["HYDROTOOLS_TMP_DIR"]
    if os.environ.get("HYDROTOOLS_TMP_DIR", None) is not None
    else gettempdir()
)

GRASS_TMP: str = (
    os.environ["HYDROTOOLS_TMP_DIR"]
    if os.environ.get("HYDROTOOLS_TMP_DIR", None) is not None
    else gettempdir()
)

GRASS_FLAGS: dict = {"quiet": False}

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
