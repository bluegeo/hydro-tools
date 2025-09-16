import os
import shutil
from typing import List, Tuple, Union
import warnings
from tempfile import _get_candidate_names
from subprocess import run

import numpy as np
import dask.array as da
from grass_session import Session
from pyproj import Transformer, CRS
from scipy.ndimage import distance_transform_edt

# Implicitly becomes available after importing grass_session
from grass.script import core as grass  # noqa
from grass.pygrass.modules.shortcuts import raster as graster  # noqa
from grass.script import array as garray  # noqa

from hydrotools.config import TMP_DIR, GRASS_TMP, GRASS_FLAGS, COG_ARGS


warnings.filterwarnings("ignore")


def named_temp_file(ext):
    return os.path.join(TMP_DIR, f"{next(_get_candidate_names())}.{ext}")


class TempRasterFile:
    def __init__(self):
        self.path = named_temp_file("tif")

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.isfile(self.path):
            os.remove(self.path)


class TempRasterFiles:
    def __init__(self, num: int):
        self.paths = [named_temp_file("tif") for _ in range(num)]

    def __enter__(self) -> list:
        return self.paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        for path in self.paths:
            if os.path.isfile(path):
                os.remove(path)


class TempVectorFile:
    def __init__(self):
        self.path = named_temp_file("gpkg")

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.isfile(self.path):
            os.remove(self.path)


class TempVectorFiles:
    def __init__(self, num: int):
        self.paths = [named_temp_file("gpkg") for _ in range(num)]

    def __enter__(self) -> list:
        return self.paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        for path in self.paths:
            if os.path.isfile(path):
                os.remove(path)


def translate_to_cog(src: str, dst: str):
    try:
        run(["gdal_edit.py", "-stats", src], check=True)
    except:
        print("Warning: no valid pixels found")

    run(
        ["gdal_translate"]
        + COG_ARGS
        + [
            src,
            dst,
        ],
        check=True,
    )


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
        self.grass_location = f"hydro-tools-grass-{next(_get_candidate_names())}"

    def __enter__(self):
        super().__init__(
            gisdb=GRASS_TMP, location=self.grass_location, create_opts=self.dataset
        )
        super().__enter__()
        return self

    def __exit__(self, exception_type, exception, traceback):
        super().__exit__(exception_type, exception, traceback)
        try:
            shutil.rmtree(os.path.join(GRASS_TMP, self.grass_location))
        except:
            print("Warning: unable to remove grass env")

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
            ds_kwargs = {"input": dataset, "output": key}
            ds_kwargs.update(GRASS_FLAGS)

            if _type.lower() == "vector":
                grass.run_command("v.in.ogr", **ds_kwargs)
            elif _type.lower() == "raster":
                graster.external(**ds_kwargs)
            else:
                raise ValueError(f"Unknown data format {_type}")

        kwargs.update(GRASS_FLAGS)

        grass.run_command(cmd, **kwargs)

    def save_raster(self, dataset: str, out_path: str, **kwargs):
        """Save a dataset in the GRASS env to a GeoTiff

        Args:
            dataset (str): GRASS dataset name
            out_path (str): Output .tif path
        """
        if not out_path.lower().endswith(".tif"):
            out_path += ".tif"

        as_cog = kwargs.pop("as_cog", True)

        kwargs.update(GRASS_FLAGS)
        kwargs.update(
            {
                "format": "GTiff",
                "createopt": [
                    "BIGTIFF=YES",
                    "TILED=YES",
                    "COMPRESS=LZW",
                    "BLOCKXSIZE=512",
                    "BLOCKYSiZE=512",
                ],
                "flags": "c",
            }
        )

        if as_cog:
            with TempRasterFile() as tmp_dst:
                graster.out_gdal(
                    dataset,
                    output=tmp_dst,
                    **kwargs,
                )

                translate_to_cog(tmp_dst, out_path)
        else:
            graster.out_gdal(
                dataset,
                output=out_path,
                **kwargs,
            )

            run(["gdal_edit.py", "-stats", out_path], check=True)

    def save_vector(self, dataset: str, out_path: str, layer: str = None):
        """Save a dataset to a geopackage

        Args:
            dataset (str): GRASS dataset name
            out_path (str): Output .gpkg path
            layer (str): Layer name to export. Defaults to None.
        """
        if not out_path.lower().endswith(".gpkg"):
            out_path += ".gpkg"

        kwargs = {}
        if layer is not None:
            kwargs["layer"] = layer

        # Save to the output
        grass.run_command(
            "v.out.ogr",
            input=dataset,
            output=out_path,
            format="GPKG",
            overwrite=True,
            **GRASS_FLAGS,
        )


def add_grass_extension(extension_name: str):
    """Add a GRASS extension

    Args:
        extension_name (str): [description]
    """
    with GrassRunner("EPSG:4326") as gs:
        grass.run_command("g.extension", extension=extension_name, operation="add")


def infer_nodata(
    a: Union[np.ndarray, da.Array, str, float, int, bool],
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


def proj4_string(sr: Union[str, int]) -> str:
    """Collect a PROJ 4 string representation of a spatial reference input. This is
    primarily for `fiona`.

    Args:
        sr (Union[str, int]): Input spatial reference

    Returns:
        str: PROJ 4 string
    """
    sr_data = CRS.from_user_input(sr)

    return sr_data.to_proj4()


def compare_projections(proj1, proj2):
    return CRS.from_user_input(proj1) == CRS.from_user_input(proj2)


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

    points = np.asarray(points)

    t_points = transformer.transform(points[:, 0], points[:, 1])

    return list(zip(*t_points))


def kernel_from_distance(distance: float, csx: float, csy: float) -> np.ndarray:
    """
    Calculate a kernel mask using distance.

    :param distance (float): Radius for kernel.
    :param csx (float): Cell size in the x-direction.
    :param csy (float): Cell size in the y-direction.
    :return (np.ndarray): Kernel mask.
    """
    num_cells_x = np.ceil(round((distance * 2.0) / csx)) + 1
    num_cells_y = np.ceil(round((distance * 2.0) / csy)) + 1

    centroid = (int((num_cells_y - 1) / 2.0), int((num_cells_x - 1) / 2.0))

    kernel = np.ones(shape=(int(num_cells_y), int(num_cells_x)), dtype=bool)
    kernel[centroid] = 0

    dt = distance_transform_edt(kernel, (csy, csx))

    return dt <= distance


from subprocess import run

run("git clone https://github.com/bluegeo/fast-watershed.git")