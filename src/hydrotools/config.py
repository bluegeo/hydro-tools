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

CACHE_DIR: str = (
    os.environ["HYDROTOOLS_CACHE_DIR"]
    if os.environ.get("HYDROTOOLS_CACHE_DIR", None) is not None
    else os.path.join(os.path.expanduser("~"), ".cache", "hydrotools")
)

GRASS_FLAGS: dict = {"quiet": True}

#: Compression for every GeoTIFF hydro-tools writes. Paired with a predictor
#: (see `predictor`), ZSTD is lossless and far smaller than LZW on elevation and
#: other continuous surfaces: on float32 DEM tiles, 3.5-4.3x smaller than LZW,
#: which managed only 13-22% over uncompressed. Needs GDAL built with libzstd
#: (standard in GDAL >= 2.3 builds, rasterio wheels and the osgeo/gdal images).
COMPRESSION = "ZSTD"


def predictor(dtype) -> int:
    """The GeoTIFF predictor for a data type.

    3 (floating point) for floats and 2 (horizontal differencing) for integers.
    A predictor stores each value as its difference from the previous one, which
    on a smooth surface leaves small, repetitive numbers for the compressor. GDAL
    refuses 3 for integer data, so it must follow the type being written.

    Args:
        dtype: A numpy dtype or name (`"float32"`), or a GDAL type name
            (`"Float32"`, `"Byte"`).
    """
    name = str(getattr(dtype, "name", dtype)).lower()
    return 3 if name.startswith("float") else 2


def gtiff_args(dtype) -> list:
    """Command-line arguments for a tiled, compressed GeoTIFF of `dtype`, for
    `gdalwarp`, `gdaldem` and similar tools."""
    return [
        "-of",
        "GTiff",
        "-co",
        f"COMPRESS={COMPRESSION}",
        "-co",
        f"PREDICTOR={predictor(dtype)}",
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


#: As `gtiff_args`, without a predictor, so it is valid for any data type. Kept
#: for callers that do not know the type they write; prefer `gtiff_args`.
GDALWARP_ARGS = [
    "-of",
    "GTiff",
    "-co",
    f"COMPRESS={COMPRESSION}",
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

#: The COG driver chooses the predictor itself with `PREDICTOR=YES`: 2 for
#: integer types, 3 for floating point.
COG_ARGS = [
    "-of",
    "COG",
    "-co",
    f"COMPRESS={COMPRESSION}",
    "-co",
    "PREDICTOR=YES",
    "-co",
    "BIGTIFF=YES",
    "-q",
]
