import os
from typing import Union
from contextlib import contextmanager
from subprocess import run
from tempfile import gettempdir, _get_candidate_names

import numpy as np
import dask.array as da
import rasterio
from rasterio.windows import Window

from hydrotools.utils import infer_nodata
from hydrotools.utils import GrassRunner
from hydrotools.config import CHUNKS, TMP_DIR, GDAL_DEFAULT_ARGS


def warp(
    source: Union[str, list, tuple],
    destination: str,
    bounds: Union[None, list, tuple] = None,
    csx: Union[None, float] = None,
    csy: Union[None, float] = None,
    t_srs: Union[None, str, int] = None,
    dtype: str = "Float32",
    resample_method: str = "cubicspline",
    overviews: bool = False,
):
    """Warp a source or sources into a destination using a resolution and extent

    Args:
        source (Union[str, list, tuple]): Raster(s) to warp
        destination (str): Destination for warped raster
        bounds (Union[list, tuple]): (left, bottom, right, top). Defaults to None.
        csx (float): Cell size in the x-direction. Defaults to None.
        csy (float): Cell size in the y-direction. Defaults to None.
        t_srs (Union[str, int]): Target spatial reference. Defaults to None.
        dtype (str, optional): Target data type. Defaults to "Float32".
        resample_method (str, optional): Resample method. Defaults to "cubicspline".
        overviews (bool, optional): Add overviews to destination. Defaults to False.
    """
    # Ensure source is a list
    if not isinstance(source, (tuple, list)):
        source = [source]
    else:
        source = list(source)

    if csx is None and csy is not None:
        csx = csy
    if csy is None and csx is not None:
        csy = csx

    cmd = (
        ["gdalwarp"]
        + ([] if bounds is None else ["-te"] + list(map(str, bounds)))
        + ([] if csx is None else ["-tr", str(csx), str(csy)])
        + ([] if t_srs is None else ["-t_srs", t_srs])
        + [
            "-r",
            resample_method,
            "-ot",
            dtype,
        ]
        + GDAL_DEFAULT_ARGS
        + source
        + [destination]
    )

    run(cmd, check=True)

    if overviews:
        cmd = ["gdaladdo", destination]
        run(cmd, check=True)


def warp_like(
    source: Union[str, list, tuple],
    destination: str,
    template: str,
    resample_method: str = "cubicspline",
    overviews: bool = False,
    **kwargs,
):
    """Warp a raster into a destination using a template raster

    Args:
        source (Union[str, list, tuple]): Source raster(s) to warp
        destination (str): Output location for warped raster
        template (str): Raster to use to define the spatial parameters for the warp

    kwargs:
        Override any parameters compatible with `warp`
    """
    rs = Raster(template)

    warp(
        source,
        destination,
        (
            kwargs.get("left", rs.left),
            kwargs.get("bottom", rs.bottom),
            kwargs.get("right", rs.right),
            kwargs.get("top", rs.top),
        ),
        kwargs.get("csx", rs.csx),
        kwargs.get("csy", rs.csy),
        kwargs.get("wkt", rs.wkt),
        kwargs.get("dtype", rs.dtype),
        resample_method,
        overviews,
    )


def vectorize(raster_source, destination, smooth_corners=False):
    if smooth_corners:
        # GRASS implementation
        with GrassRunner(raster_source) as gr:
            gr.run_command(
                "r.to.vect",
                (raster_source, "rast", "raster"),
                input="rast",
                output="vect",
                type="area",
                flags="s",
            )
            gr.save_vector("vect", destination)

    else:
        cmd = ["gdal_polygonize.py", "-8", raster_source, destination]
        run(cmd, check=True)


class Raster:
    """Abstract raster data using dask"""

    def __init__(self, src: str):
        """Open a raster

        Args:
            src (str): Path to a GDAL-supported raster
        """
        self.__dict__.update(self.raster_specs(src))
        self.src = src

    @staticmethod
    def raster_specs(src: str) -> dict:
        """Collect raster specifications.

        Note, the data type and no data value are interpreted from the first band only

        Args:
            src (str): Path to a GDAL-supported raster

        Returns:
            (dict): Raster properties
        """
        with rasterio.open(src) as ds:
            left, bottom, right, top = list(ds.bounds)
            csx, csy = ds.res
            band_count = ds.count
            shape = (band_count, ds.height, ds.width)

            return dict(
                band_count=band_count,
                wkt=ds.crs.wkt,
                left=left,
                csx=csx,
                top=top,
                csy=csy,
                shape=shape,
                bottom=bottom,
                right=right,
                nodata=ds.nodatavals[0],
                bbox=(left, bottom, right, top),
                dtype=ds.dtypes[0],
                ndim=3,
            )

    @classmethod
    def empty(
        cls,
        destination: str,
        top: float,
        left: float,
        csx: float,
        csy: float,
        crs: str,
        shape: tuple,
        dtype: str,
        nodata: Union[int, float, bool, None] = None,
    ):
        """Create an empty raster

        Args:
            top (float): Top coordinate
            left (float): Left coordinate
            csx (float): Cell size in the x-direction
            csy (float): Cell size in the y-direction
            crs (str): Coordinate reference system
            shape (tuple): Shape (first dimension must be number of bands)
            dtype (str): Data type in numpy string form
            nodata (Union[int, float, bool, None]): Specific no data value. Defaults
            to None.
        """
        if nodata is None:
            nodata = infer_nodata(dtype)

        transform = rasterio.transform.from_origin(
            float(left), float(top), float(csx), float(csy)
        )

        new_dataset = rasterio.open(
            destination,
            "w",
            driver="GTiff",
            count=shape[0],
            height=shape[1],
            width=shape[2],
            transform=transform,
            crs=crs,
            dtype=dtype,
            nodata=nodata,
            # -co
            tiled=True,
            compress="LZW",
            blockxsize=512,
            blockysize=512,
            bigtiff=True,
        )

        for band in range(shape[0]):
            for _, window in new_dataset.block_windows(band + 1):
                a = np.full((window.height, window.width), nodata, dtype)
                new_dataset.write(a, indexes=band + 1, window=window)

        new_dataset.close()

        return cls(destination)

    @classmethod
    def empty_like(cls, destination: str, template: str, **kwargs):
        """Create an empty raster using a template raster

        Args:
            destination (str): Destination raster path
            template (Union[Raster, str]): Template raster path

        Kwargs:
            Any of the raster specs may be overridden with kwargs

        Returns:
            Raster: Populated empty raster instance
        """
        if isinstance(template, str):
            specs = cls.raster_specs(template)
        else:
            specs = template

        return cls.empty(
            destination,
            kwargs.get("top", specs["top"]),
            kwargs.get("left", specs["left"]),
            kwargs.get("csx", specs["csx"]),
            kwargs.get("csy", specs["csy"]),
            kwargs.get("wkt", specs["wkt"]),
            kwargs.get("shape", specs["shape"]),
            kwargs.get("dtype", specs["dtype"]),
            kwargs.get("nodata", specs["nodata"]),
        )

    def __getitem__(self, s: Union[slice, int, tuple, None]) -> np.ndarray:
        """Collect raster data as a numpy array.
        Data are pulled from the raster source directly with the slice

        Args:
            s (Union[slice, int, tuple, None]): Slicing parameter

        Returns:
            (np.ndarray): 3D numpy array with the shape (band, y, x)
        """
        if s == (slice(0, 0, None), slice(0, 0, None), slice(0, 0, None)):
            # Dask calls with empty slices when using from_array
            return np.array([]).reshape((0, 0))

        # Collect a compatible slice
        bands, col_off, row_off, width, height = self.parse_slice(s)

        # Allocate output
        out_array = np.empty((len(bands), height, width), self.dtype)

        with self.ds() as ds:
            window = Window(col_off, row_off, width, height)
            for i, band in enumerate(bands):
                out_array[i, ...] = ds.read(int(band + 1), window=window)

        return out_array

    def __setitem__(self, s: Union[slice, int, tuple, None], a: np.ndarray):
        """Place array data in the raster

        Args:
            s (Union[slice, int, tuple, None]): Slicing parameter
            a (np.ndarray): numpy array
        """
        bands, col_off, row_off, width, height = self.parse_slice(s)

        with self.ds("r+") as ds:
            window = Window(col_off, row_off, width, height)
            for i, band in enumerate(bands):
                ds.write(a[i, ...], indexes=int(band + 1), window=window)

    def parse_slice(self, s: Union[slice, int, tuple, None]) -> tuple:
        """Prepare a slice argument

        Args:
            s (Union[slice, int, tuple, None]): Slice parameter

        Returns:
            (tuple): bands, col_off, row_off, width, height
        """

        def get_slice_item(item, dim, ints=False):
            """Collect a slice item in a compatible format"""
            if isinstance(item, int):
                if item > dim - 1:
                    raise IndexError(
                        f"Index {item} out for bounds for dimension of size {dim}"
                    )
                if ints:
                    return [item]
                else:
                    return item, 1
            elif isinstance(item, slice):
                if item.stop is None:
                    stop = dim
                else:
                    stop = item.stop

                if item.start is None:
                    start = 0
                else:
                    start = item.start

                if stop > dim:
                    raise IndexError(
                        f"Index {stop} out for bounds for dimension of size {dim}"
                    )
                if ints:
                    return np.arange(start, stop, item.step)
                else:
                    return start, stop - start
            elif item is None:
                if ints:
                    return np.arange(0, dim)
                else:
                    return 0, dim
            elif isinstance(item, list):
                if max(item) > dim - 1:
                    raise IndexError(
                        f"Index {max(item)} out for bounds for dimension of size {dim}"
                    )
                if ints:
                    return item
                else:
                    raise NotImplementedError(
                        "Integer indexing not supported for this dimension"
                    )
            else:
                raise NotImplementedError(
                    f"Slices of type {type(item).__name__} are not supported"
                )

        if not hasattr(s, "__iter__"):
            s = [s]

        if len(s) > 3:
            raise IndexError("Rasters must be indexed in a maximum of 3 dimensions")

        # Ensure s includes 3 dimensions
        s = list(s)
        s += [None] * (3 - len(s))

        # Collect bands as a list of ints
        bands = get_slice_item(s[0], self.band_count, True)

        # Collect remaining two dimensions
        row_off, height = get_slice_item(s[1], self.shape[1])
        col_off, width = get_slice_item(s[2], self.shape[2])

        if max(bands) > self.band_count:
            raise IndexError(
                f"Band index {max(bands)} out of bounds with band count {self.band_count}"
            )
        if row_off > self.shape[1] - 1:
            raise IndexError(
                f"Index {row_off} out of bounds for axis 0 with shape {self.shape[1]}"
            )
        if col_off > self.shape[2] - 1:
            raise IndexError(
                f"Index {col_off} out of bounds for axis 1 with shape {self.shape[2]}"
            )

        return bands, col_off, row_off, width, height

    @contextmanager
    def ds(self, mode="r") -> rasterio.DatasetReader:
        """Open the underlying raster dataset

        Returns:
            (rasterio.DatasetReader): Dataset for array collection
        """
        ds = rasterio.open(self.src, mode)
        yield ds
        ds.close()

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure

        figure(figsize=(8, 6), dpi=80)

        a = np.ma.masked_equal(np.squeeze(self[:]), self.nodata)

        plt.imshow(a)
        plt.show()


def from_raster(src: Union[Raster, str], chunks: tuple = CHUNKS) -> da.Array:
    """Read a raster dataset into a dask array masked by the no data value

    Args:
        src (Union[Raster, str]): Source GDAL-supported raster dataset
        chunks (tuple, optional): Dask chunks. Defaults to CHUNKS.

    Returns:
        (da.Array): Masked dask array in the shape (bands, rows, cols)
    """
    if not isinstance(src, Raster):
        src = Raster(src)

    a = da.from_array(src, chunks=chunks)

    a = da.ma.masked_where((a == src.nodata) | da.isnan(a), a)
    da.ma.set_fill_value(a, infer_nodata(a))

    return a


def to_raster(data: da.Array, template: str, destination: str, overviews: bool = True):
    """Compute a dask array and store it in a destination raster dataset

    Args:
        data (da.Array): Array of data to compute and store in the output raster
        template (str): Raster to use for specifications
        destination (str): Output raster path
        overviews (bool): Build overviews. Defaults to True.
    """
    # The nodata value should be inferred from the data type and filled
    nodata = infer_nodata(data)
    dtype = data.dtype.name
    da.ma.set_fill_value(data, nodata)

    output_raster = Raster.empty_like(destination, template, dtype=dtype, nodata=nodata)

    da.store([da.ma.filled(data)], [output_raster])

    if overviews:
        cmd = ["gdaladdo", destination]
        run(cmd, check=True)


def raster_where(
    cond: da.Array,
    if_true: Union[da.Array, int, float, bool],
    if_false: Union[da.Array, int, float, bool],
) -> da.Array:
    """Perform a remapping on a raster array while preserving a nodata mask

    Args:
        cond (da.Array): Condition
        if_true (Union[da.Array, int, float, bool]): Value if True
        if_false (Union[da.Array, int, float, bool]): Value if False
    """
    a = da.where(cond, if_true, if_false)

    # Remask
    if_true_mask = da.ma.getmaskarray(if_true)
    if if_true_mask.shape:
        # An array with or without a mask - use the mask where it meets the condition
        if_true_mask = cond & if_true_mask
    else:
        # A scalar - do not mask any locations
        if_true_mask = da.zeros(a.shape, "bool", chunks=a.chunksize)

    if_false_mask = da.ma.getmaskarray(if_false)
    if if_false_mask.shape:
        if_false_mask = ~cond & if_false_mask
    else:
        if_false_mask = da.zeros(a.shape, dtype="bool", chunks=a.chunksize)

    a = da.ma.masked_where(if_true_mask | if_false_mask, a)
    da.ma.set_fill_value(a, infer_nodata(a))

    return a


class TempRasterFile:
    def __init__(self):
        self.path = os.path.join(TMP_DIR, next(_get_candidate_names()) + ".tif")

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.isfile(self.path):
            os.remove(self.path)
