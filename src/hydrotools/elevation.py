from subprocess import run
from typing import Union

import numpy as np
import dask.array as da
from scipy.ndimage import binary_erosion

from hydrotools.utils import (
    GrassRunner,
    TempRasterFile,
    TempRasterFiles,
    translate_to_cog,
)
from hydrotools.raster import (
    from_raster,
    raster_where,
    to_raster,
    warp_like,
)
from hydrotools.config import CHUNKS, GDALWARP_ARGS
from hydrotools.interpolate import PointInterpolator


def slope(
    dem: str,
    destination: str,
    units: str = "degrees",
    scale: float = 1,
):
    """Calculate topographic slope.
    See [gdaldem](https://gdal.org/programs/gdaldem.html#slope).

    Args:
        dem (str): Digital Elevation model raster
        destination (str): Output slope raster destination
        units (str, optional): Units for the output - choose one of
        [`"degrees"`, `"percent"`]. Defaults to "degrees".
        scale (float, optional): Z-factor to scale the output. Defaults to 1.
    """
    with TempRasterFile() as tmp_dst:
        cmd = [
            "gdaldem",
            "slope",
            "-compute_edges",
            "-s",
            str(scale),
            dem,
            tmp_dst,
        ] + GDALWARP_ARGS

        if units.lower() == "percent":
            cmd += ["-p"]

        run(cmd, check=True)

        translate_to_cog(tmp_dst, destination)


def aspect(
    dem: str,
    destination: str,
):
    """Calculate aspect. See [gdaldem](https://gdal.org/programs/gdaldem.html#aspect).

    Args:
        dem (str): Digital Elevation model raster
        destination (str): Output aspect raster destination
    """
    with TempRasterFile() as tmp_dst:
        cmd = [
            "gdaldem",
            "aspect",
            "-compute_edges",
            dem,
            tmp_dst,
        ] + GDALWARP_ARGS

        run(cmd, check=True)

        translate_to_cog(tmp_dst, destination)


def solar_radiation(dem: str, day: int, step: float, rad_dst: str):
    """Calculate a solar radiation grid for a given day.

    Args:
        dem (str): Digital Elevation Model.
        day (int): Day (1-365).
        step (float): Time step in decimal hours when computing all-day radiation sums.
        rad_dst (str): Output irradiance/irradiation raster in W/m**2.
    """
    if day < 1 or day > 365:
        raise ValueError("Invalid day for 1 <= day <= 365")
    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.sun",
            (dem, "dem", "raster"),
            elevation="dem",
            day=int(day),
            step=float(step),
            glob_rad="rad",
        )
        gr.save_raster("rad", rad_dst)


def terrain_ruggedness_index(
    dem: str,
    destination: str,
):
    """Calculate Terrain Ruggedness Index (TRI).
    See [gdaldem](https://gdal.org/programs/gdaldem.html#tri).

    Args:
        dem (str): Digital Elevation model raster
        destination (str): Output TRI raster destination
    """
    with TempRasterFile() as tmp_dst:
        cmd = [
            "gdaldem",
            "TRI",
            "-compute_edges",
            dem,
            tmp_dst,
        ] + GDALWARP_ARGS

        run(cmd, check=True)

        translate_to_cog(tmp_dst, destination)


def z_align(
    source: str,
    addition: str,
    destination: str,
    resample_interpolation: Union[float, None] = None,
    **kwargs,
):
    """Merge rasters and align z-values using overlapping areas.

    The `addition` will be modified and added to `source`, and saved to the `destination`

    Args:
        source (str): Source raster
        addition (str): Raster to be aligned and added
        destination (str): Location of output raster
        resample_interpolation (Union[float, None]): Isotropic resample resolution for
        the interpolation of the delta of overlapping areas. This is an expensive
        operation and can be approximated for larger datasets by
        resampling the delta to a lower resolution and returning it to the original
        resolution using cubic interpolation.
    """
    # Load dasks
    source_a = from_raster(source)
    addition_a = from_raster(addition)

    if addition_a.shape != source_a.shape:
        raise ValueError(
            "Source and Addition rasters must spatially align. Use `warp_like` first"
        )

    # Compute a delta
    delta = source_a - addition_a

    # Area to be added - where the source has no data and the addition does
    fill_area = da.ma.getmaskarray(source_a) & ~da.ma.getmaskarray(addition_a)

    def interp(obs: da.Array, pred: da.Array, n_neighbours: int) -> np.ndarray:
        """Perform interpolation

        Args:
            obs (da.Array): Observed data
            pred (da.Array): Predicted points (must be boolean)
            n_neighbours (int): Number of neighbours to evaluate for IDW.

        Returns:
            np.ndarray
        """
        obs, pred = da.compute(obs, pred)

        obs_mask = ~np.ma.getmaskarray(obs)

        # Collect the edges of overlapping area between the source and addition
        overlap = ~binary_erosion(obs_mask, np.ones((1, 3, 3), bool)) & obs_mask

        pred_z = PointInterpolator(
            np.vstack(np.where(overlap)).T[:, 1:],
            obs[overlap].data,
            np.vstack(np.where(pred)).T[:, 1:],
        ).idw(n_neighbours)

        nodata = np.finfo("float32").min
        output = np.full(pred.shape, nodata, "float32")
        output[pred] = pred_z

        return da.from_array(
            np.ma.masked_values(output, nodata, copy=False), chunks=CHUNKS
        )

    # Add interpolated deltas to the delta array
    with TempRasterFiles(7) as (
        obs_tmp,
        pred_tmp,
        obs_dst,
        pred_dst,
        delta_interp,
        delta_tmp,
        delta_dst,
    ):
        if resample_interpolation is not None:
            to_raster(delta, source, obs_tmp, as_cog=False)
            warp_like(
                obs_tmp,
                obs_dst,
                source,
                csx=resample_interpolation,
                csy=resample_interpolation,
            )

            # Fill area must be masked to reflect nodata, and changed from bool
            to_raster(
                da.ma.masked_where(~fill_area, fill_area).astype("uint8"),
                source,
                pred_tmp,
                as_cog=False,
            )
            warp_like(
                pred_tmp,
                pred_dst,
                source,
                csx=resample_interpolation,
                csy=resample_interpolation,
            )

            delta_a = interp(
                from_raster(obs_dst),
                ~da.ma.getmaskarray(from_raster(pred_dst)),
                kwargs.get("n_neighbours", 1000),
            )

            to_raster(delta_a, obs_dst, delta_interp, as_cog=False)

            # Expand to ensure no gaps result from resampling
            with GrassRunner(delta_interp) as gr:
                gr.run_command(
                    "r.fill.stats",
                    (delta_interp, "r", "raster"),
                    input="r",
                    output="e",
                    distance=2,
                    flags="ks",
                )
                gr.save_raster("e", delta_tmp)

            warp_like(delta_tmp, delta_dst, source)

            delta = from_raster(delta_dst)

        else:
            delta = interp(
                delta,
                fill_area,
                kwargs.get("n_neighbours", 1000),
            )

        output = raster_where(~da.ma.getmaskarray(delta), addition_a + delta, source_a)

        to_raster(output, source, destination)


def las_to_dtm(
    las_path: str,
    destination: str,
    top: float,
    bottom: float,
    left: float,
    right: float,
    epsg: int,
    csx: float,
    csy: float,
):
    """Generate a Digital Terrain Model using a .las file. If data are provided in an
    ascii format such as .xyz, they should be converted to .las. For example:

    ```
    txt2las -i lidar.xyz -o lidar.las
    ```

    Args:
        las_path (str): [description]
        destination (str): [description]
        top (float): [description]
        bottom (float): [description]
        left (float): [description]
        right (float): [description]
        epsg (int): [description]
        csx (float): [description]
        csy (float): [description]
    """
    with GrassRunner(f"EPSG:{epsg}") as gs:
        # Set the region
        gs.run_command(
            "g.region", n=top, s=bottom, w=left, e=right, nsres=csy, ewres=csx
        )

        # import
        gs.run_command("v.in.lidar", input=las_path, output="points")

        # detection
        gs.run_command("v.lidar.edgedetection", input="points", output="edge")
        gs.run_command(
            "v.lidar.growing", input="edge", output="growing", first="points_first"
        )
        gs.run_command(
            "v.lidar.correction",
            input="growing",
            output="correction",
            terrain="only_terrain",
        )

        # interpolation
        gs.run_command("v.surf.rst", input="only_terrain", elevation="terrain")

        # save output
        gs.save_raster("terrain", destination)
