import os
import shutil
from tempfile import gettempdir
from typing import List, Tuple, Union
import warnings
from subprocess import run

import numpy as np
import dask.array as da
from grass_session import Session
from pyproj import Transformer

# Implicitly becomes available after importing grass_session
from grass.script import core as grass  # noqa
from grass.pygrass.modules.shortcuts import raster as graster  # noqa

from hydrotools.config import GRASS_TMP, GRASS_LOCATION, GDAL_DEFAULT_ARGS


warnings.filterwarnings("ignore")


class GrassRunner(Session):
    """Run GRASS GIS commands using the grass-session library and simplified
    control over mapset and external dataset abstraction.
    """

    def __init__(self, dataset: str):
        """Initialize GRASS with a dataset. The dataset may be as simple as a spatial
        reference, or a vector/raster dataset. When using vector -> raster commands
        it is useful to initialize with a raster dataset, which will serve as the
        environment for all resulting data.

        Args:
            dataset (str): GRASS mapset initialization data.
        """
        self.dataset = dataset

    def __enter__(self):
        super().__init__(
            gisdb=GRASS_TMP, location=GRASS_LOCATION, create_opts=self.dataset
        )
        super().__enter__()
        return self

    def __exit__(self, exception_type, exception, traceback):
        super().__exit__(exception_type, exception, traceback)
        shutil.rmtree(os.path.join(GRASS_TMP, GRASS_LOCATION))

    def run_command(self, cmd: str, *args, **kwargs):
        """Run a grass command

        Args:
            cmd (str): Grass command. Example `r.watershed`.
            args (tuple): External data to use within the GRASS comand. This is a
            3-tuple with the following form:
                ("/path/to/data...", "name", "vector | raster | None").
            The "name" attribute is used for GRASS inputs as kwargs, as described in the
            documentation.
            kwargs (dict): Arguments for the grass command, for example:
                https://grass.osgeo.org/grass76/manuals/r.watershed.html

        """
        for dataset, key, _type in args:
            if _type.lower() == "vector":
                grass.run_command("v.in.ogr", input=dataset, output=key)
            elif _type.lower() == "raster":
                graster.external(input=dataset, output=key)
            else:
                raise ValueError(f"Unknown data format {_type}")

        grass.run_command(cmd, **kwargs)

    def save_raster(self, dataset: str, out_path: str, **kwargs):
        """Save a dataset in the GRASS env to a COG

        Args:
            dataset (str): GRASS dataset name
            out_path (str): Output .tif path
        """
        if not out_path.lower().endswith(".tif"):
            out_path += ".tif"

        graster.out_gdal(
            dataset,
            format="GTiff",
            output=out_path,
            createopt=[
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512",
                "COMPRESS=LZW",
                "BIGTIFF=YES",
            ],
            **kwargs,
        )

    def save_vector(self, dataset: str, out_path: str):
        """Save a dataset to a geopackage

        Args:
            dataset (str): GRASS dataset name
            out_path (str): Output .gpkg path
        """
        if not out_path.lower().endswith(".gpkg"):
            out_path += ".gpkg"

        # Save to the output
        grass.run_command(
            "v.out.ogr", input=dataset, output=out_path, format="GPKG", overwrite=True
        )


def add_grass_extension(extension_name: str):
    """Add a GRASS extension

    Args:
        extension_name (str): [description]
    """
    with GrassRunner("EPSG:4326") as gs:
        grass.run_command(
            "g.extension", extension=extension_name, operation="add"
        )


def infer_nodata(
    a: Union[np.ndarray, da.Array, str, float, int, bool]
) -> Union[float, int, bool]:
    """Collect a nodata value based on a dataset's data type

    Args:
        a (Union[np.ndarray, da.Array, str, float, int, bool]): Array or data type name

    Returns:
        Union[float, int, bool]: Suggested no data value for data type
    """
    dtypes = {
        "float64": np.finfo("float64").min,
        "float32": np.finfo("float32").min,
        "int64": np.iinfo("int64").min,
        "uint64": np.iinfo("uint64").max,
        "int32": np.iinfo("int32").min,
        "uint32": np.iinfo("uint32").max,
        "int16": np.iinfo("int16").min,
        "uint16": np.iinfo("uint16").max,
        "int8": np.iinfo("int8").min,
        "uint8": np.iinfo("uint8").max,
        "bool": False,
    }

    if hasattr(a, "dtype"):
        return dtypes[a.dtype.name]
    else:
        try:
            return dtypes[a]
        except KeyError:
            try:
                return dtypes[np.array(a).dtype.name]
            except:
                raise ValueError(f"Unable to collect data type from object {a}")


def indices_to_coords(
    indices: Tuple[Union[np.ndarray, list], Union[np.ndarray, list]],
    top: float,
    left: float,
    csx: float,
    csy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a tuple of indices to geographic coordinates

    Args:
        indices (Tuple[Union[np.ndarray, list], Union[np.ndarray, list]]): Coordinates
        in the form: `([i...], [j...])`
        top (float): Top corner coordinate
        left (float): Left corner coordinate
        csx (float): Cell size in the x-direction
        csy (float): Cell size in the y-direction

    Returns:
        Tuple[np.ndarray, np.ndarray]: Block-centered coordinates in the form
        `([y...], [x...])`
    """
    i, j = np.asarray(indices[0]), np.asarray(indices[1])

    return ((top - (csy / 2.0)) - (i * csy), (left + (csx / 2.0)) + (j * csx))


def transform_points(
    points: List[Tuple[float, float]],
    in_proj: Union[int, str],
    out_proj: Union[int, str],
) -> List[Tuple[float, float]]:
    """Transform a list of points

    Args:
        points (List[Tuple[float, float]]): Points in the form `[(x, y), (x, y)...]`
        in_proj (Union[int, str]): Input coordinate reference system
        out_proj (Union[int, str]): Output coordinate reference system

    Returns:
        List[Tuple[float, float]]: Transformed points in the form `[(x, y), (x, y)...]`
    """
    transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)

    points = np.array(points)

    t_points = transformer.transform(points[:, 0], points[:, 1])

    return list(zip(*t_points))
