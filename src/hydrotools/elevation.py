from subprocess import run
from typing import Union

import numpy as np
import dask.array as da
from scipy.ndimage import binary_erosion

from hydrotools.raster import (
    TempRasterFile,
    from_raster,
    raster_where,
    to_raster,
    warp_like,
)
from hydrotools.config import CHUNKS, GDAL_DEFAULT_ARGS
from hydrotools.interpolate import PointInterpolator


def slope(
    dem: str,
    destination: str,
    units: str = "degrees",
    scale: float = 1,
    overviews: bool = True,
):
    """Calculate topographic slope

    Args:
        dem (str): Digital Elevation model raster
        destination (str): Output slope raster destination
        units (str, optional): Units for the output. Defaults to "degrees".
        scale (float, optional): Z-factor to scale the output. Defaults to 1.
        overviews (bool, optional): Build overviews for the output. Defaults to True.
    """
    gdal_args = [
        arg
        for arg in GDAL_DEFAULT_ARGS
        if arg not in ["-multi", "-wo", "NUM_THREADS=ALL_CPUS"]
    ]

    cmd = [
        "gdaldem",
        "slope",
        "-s",
        str(scale),
        dem,
        destination,
    ] + gdal_args

    if units.lower() == "percent":
        cmd += ["-p"]

    run(cmd, check=True)

    if overviews:
        cmd = ["gdaladdo", destination]
        run(cmd, check=True)


def z_align(
    source: str,
    addition: str,
    destination: str,
    resample_interpolation: Union[float, None] = None,
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
        resolution using a cubic spline.
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

    def interp(obs: da.Array, pred: da.Array, n_neighbours: int = 10000) -> np.ndarray:
        """Perform interpolation

        Args:
            obs (da.Array): Observed data
            pred (da.Array): Predicted points (must be boolean)
            n_neighbours (int): Number of neighbours to evaluate for IDW.
            Defaults to 10000.

        Returns:
            np.ndarray
        """
        obs, pred = da.compute(obs, pred)

        obs_mask = ~np.ma.getmaskarray(obs)

        # Collect the edges of overlapping area between the source and addition
        overlap = ~binary_erosion(obs_mask, np.ones((1, 3, 3), np.bool)) & obs_mask

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
    with (
        TempRasterFile() as obs_tmp,
        TempRasterFile() as pred_tmp,
        TempRasterFile() as obs_dst,
        TempRasterFile() as pred_dst,
        TempRasterFile() as delta_tmp,
        TempRasterFile() as delta_dst,
    ):
        if resample_interpolation is not None:
            to_raster(delta, source, obs_tmp, overviews=False)
            warp_like(
                obs_tmp,
                obs_dst,
                source,
                csx=resample_interpolation,
                csy=resample_interpolation,
                add_overviews=False,
            )

            # Fill area must be masked to reflect nodata, and changed from bool
            to_raster(
                da.ma.masked_where(~fill_area, fill_area).astype("uint8"),
                source,
                pred_tmp,
                overviews=False,
            )
            warp_like(
                pred_tmp,
                pred_dst,
                source,
                csx=resample_interpolation,
                csy=resample_interpolation,
                add_overviews=False,
            )

            delta_a = interp(
                from_raster(obs_dst), ~da.ma.getmaskarray(from_raster(pred_dst))
            )

            to_raster(delta_a, obs_dst, delta_tmp, overviews=False)

            warp_like(delta_tmp, delta_dst, source, add_overviews=False)

            delta = from_raster(delta_dst)

            # TODO: During resampling, some fill area may end up as no data
        else:
            delta = interp(delta, fill_area)

        output = raster_where(~da.ma.getmaskarray(delta), addition_a + delta, source_a)

        to_raster(output, source, destination)
