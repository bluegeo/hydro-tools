from typing import Union, Tuple, Generator, List
import os
from shutil import rmtree
from contextlib import contextmanager
from subprocess import run
from tempfile import _get_candidate_names

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

from hydrotools.utils import infer_nodata, translate_to_cog, compare_projections
from hydrotools.utils import GrassRunner, TempRasterFile
from hydrotools.config import CHUNKS, TMP_DIR, GDALWARP_ARGS


def warp(
    source: Union[str, list, tuple],
    destination: str,
    bounds: Union[None, list, tuple] = None,
    csx: Union[None, float] = None,
    csy: Union[None, float] = None,
    t_srs: Union[None, str, int] = None,
    dtype: str = "Float32",
    resample_method: str = "cubic",
    additional_args: list = [],
    as_cog: bool = True,
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
        resample_method (str, optional): Resample method. Defaults to "cubic".
        additional_args (list, optional): Any additional gdalwarp arguments.
        as_cog (bool, optional): Output a GeoTiff in COG format. Defaults to True.
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
            "-multi",
            "-r",
            resample_method,
            "-ot",
            "Byte" if dtype == "uint8" else dtype,
        ]
        + GDALWARP_ARGS
        + additional_args
        + source
    )

    if as_cog:
        with TempRasterFile() as tmp_dst:
            run(cmd + [tmp_dst], check=True)

            translate_to_cog(tmp_dst, destination)
    else:
        run(cmd + [destination], check=True)

        try:
            run(["gdal_edit.py", "-stats", destination], check=True)
        except:
            pass


def warp_like(
    source: Union[str, list, tuple],
    destination: str,
    template: str,
    resample_method: str = "cubic",
    as_cog: bool = True,
    **kwargs,
):
    """Warp a raster into a destination using a template raster

    Args:
        source (Union[str, list, tuple]): Source raster(s) to warp.
        destination (str): Output location for warped raster.
        template (str): Raster to use to define the spatial parameters for the warp.
        resample_method (str): GDAL-supported resampling interpolation method. Defaults
        to True.
        as_cog (bool): Create a COG on output. Defaults to True.

    kwargs:
        Override any parameters compatible with `warp`
    """
    if isinstance(source, str):
        rs_dtype = Raster(source).dtype
    else:
        # Most common data type
        dtypes, counts = np.unique(
            [Raster(s).dtype for s in source], return_counts=True
        )
        rs_dtype = dtypes[np.argmax(counts)]

    rt = Raster(template)

    warp(
        source,
        destination,
        (
            kwargs.get("left", rt.left),
            kwargs.get("bottom", rt.bottom),
            kwargs.get("right", rt.right),
            kwargs.get("top", rt.top),
        ),
        kwargs.get("csx", rt.csx),
        kwargs.get("csy", rt.csy),
        rt.wkt,
        kwargs.get("dtype", rs_dtype),
        resample_method,
        additional_args=kwargs.get("additional_args", []),
        as_cog=as_cog,
    )


class TXTFactory:
    """
    Enable writing of raster cell data as points to a text file or tiles using
    `dask.array.store`.
    """

    def __init__(
        self,
        shape: Tuple[int],
        band: int,
        top: float,
        left: float,
        csx: float,
        csy: float,
        out_txt: str,
        in_proj: Union[str, int],
        out_proj: Union[str, int] = 4326,
        tiles: bool = False,
    ):
        if len(shape) != 3:
            raise ValueError("Expected 3 dimensions")

        if not os.path.isdir(os.path.dirname(out_txt)):
            raise ValueError(f"Unrecognized path: {os.path.dirname(out_txt)}")

        self.out_txt = out_txt
        if self.out_txt.lower().endswith(".csv"):
            # Write headings
            with open(self.out_txt, "a") as f:
                f.write("x,y,value\n")
        elif tiles:
            self.tile_temp_dir = os.path.join(TMP_DIR, next(_get_candidate_names()))
            os.mkdir(self.tile_temp_dir)

        self.tiles = tiles
        self.shape = shape
        self.band = band - 1
        self.top = float(top)
        self.left = float(left)
        self.csx = float(csx)
        self.csy = float(csy)

        if compare_projections(in_proj, out_proj):
            self.transformer = lambda x, y: (x, y)
        else:
            self.transformer = Transformer.from_crs(
                in_proj, out_proj, always_xy=True
            ).transform

        # For dask
        self.ndim = 3
        self.dtype = np.float32

    def __setitem__(self, s: Union[slice, int, tuple, None], a: np.ndarray):
        """Place array data in the output text file (csv or ndjson)

        Args:
            s (Union[slice, int, tuple, None]): Slicing parameter
            a (np.ndarray): numpy array
        """
        _, col_off, row_off, _, _ = Raster.parse_slice(s, self.shape)

        a = a[self.band, ...]

        i, j = np.where(~np.ma.getmaskarray(a))
        if i.size > 0:
            y = self.top - (row_off * self.csy) - (i * self.csy) - self.csy * 0.5
            x = self.left + (col_off * self.csx) + (j * self.csx) + self.csx * 0.5

            x, y = self.transformer(x, y)

            data = np.char.mod("%f", np.vstack([x, y, a[i, j].data]).T)

            if self.out_txt.lower().endswith(".csv") or self.tiles:
                out_data = "\n".join([",".join(row) for row in data])
            else:
                json_format = '{{"type": "Feature", "properties": {{"value": {2}}}, "geometry": {{ "type": "Point", "coordinates": [{0}, {1}]}}}}'
                out_data = "\n".join([json_format.format(*row) for row in data])

            if self.tiles:
                out_dst_dir = os.path.join(TMP_DIR, next(_get_candidate_names()))
                os.mkdir(out_dst_dir)
                out_dst = os.path.join(
                    out_dst_dir, f"{os.path.basename(self.out_txt)}.csv"
                )
                # Write headings
                with open(out_dst, "a") as f:
                    f.write("x,y,value\n")
            else:
                out_dst = self.out_txt

            with open(out_dst, "a") as f:
                f.write(out_data + "\n")

            if self.tiles:
                next_tile_dir = os.path.join(
                    self.tile_temp_dir, next(_get_candidate_names())
                )

                cmd = [
                    "tippecanoe",
                    "-zg",
                    "-P",
                    "-ai",
                    "-e",
                    next_tile_dir,
                    "--drop-densest-as-needed",
                    "--extend-zooms-if-still-dropping",
                    out_dst,
                ]
                run(cmd, check=True)

                os.remove(out_dst)

    def consolidate_tiles(self):
        tiles = " ".join(
            [
                os.path.join(self.tile_temp_dir, f)
                for f in os.listdir(self.tile_temp_dir)
                if os.path.isdir(os.path.join(self.tile_temp_dir, f))
            ]
        )

        cmd = ["tile-join", "-e", self.out_txt, tiles]
        run(cmd, check=True)


def vectorize(
    raster_source: str,
    destination: str,
    geometry: str = "polygon",
    smooth_corners: bool = False,
    band: int = 1,
):
    """Create a vector dataset

    Args:
        raster_source (str): Raster path.
        destination (str): Destionation vector dataset.
        geometry (str, optional): Geometry type for output vector. Choose from:
            [
                "polygon",
                "point"
            ]
            Defaults to "polygon".
        smooth_corners (bool, optional): Round the corners of raster cells.
            Defaults to False.
        band (int): Raster band used for assignment of values. Defaults to Band 1.
    """
    if geometry == "polygon":
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
    elif geometry == "point":
        extensions = [".csv", ".geojsonl", ".ndjson", ".jsonl"]

        if not any([destination.lower().endswith(ext) for ext in extensions]):
            raise ValueError("Only .geojsond.json and .csv files supported")

        r_specs = Raster.raster_specs(raster_source)
        a = from_raster(raster_source)

        dst = TXTFactory(
            r_specs["shape"],
            band,
            r_specs["top"],
            r_specs["left"],
            r_specs["csx"],
            r_specs["csy"],
            destination,
            r_specs["wkt"],
            r_specs["wkt"],
        )

        da.store([a], [dst])

    else:
        raise NotImplementedError(f"Geometry of type '{geometry}' not implemented")


def to_vector_tiles(raster_source: str, tile_dst: str, band: int = 1):
    """Generate vector tiles of points with raster pixel values

    **Note, the tippecanoe library must be installed**

    Args:
        raster_source (str): A raster dataset.
        tile_dst (str): A directory (must not exist already) for output tiles.
        band (int, optional): Raster band value to add to the vector tiles as a "value"
        attribute. Defaults to 1.
    """
    r_specs = Raster.raster_specs(raster_source)

    dst = TXTFactory(
        r_specs["shape"],
        band,
        r_specs["top"],
        r_specs["left"],
        r_specs["csx"],
        r_specs["csy"],
        tile_dst,
        r_specs["wkt"],
        4326,
        True,
    )

    da.store([from_raster(raster_source)], [dst])

    dst.consolidate_tiles()

    # Explicitly clean up
    rmtree(dst.tile_temp_dir)


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
        """Create an empty raster.

        Note, since this is open for writing a GeoTiff is
        created, and will not include overviews.

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

        if not hasattr(shape, "__iter__") or len(shape) > 3:
            raise ValueError(f"Unsupported shape: {shape}")

        if len(shape) == 1:
            shape = (1, 1, shape[0])
        elif len(shape) == 2:
            shape = (1,) + shape

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

        # for band in range(shape[0]):
        #     for _, window in new_dataset.block_windows(band + 1):
        #         a = np.full((window.height, window.width), nodata, dtype)
        #         new_dataset.write(a, indexes=band + 1, window=window)

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
        bands, col_off, row_off, width, height = self.slice_params(s)

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
        bands, col_off, row_off, width, height = self.slice_params(s)

        with self.ds("r+") as ds:
            window = Window(col_off, row_off, width, height)
            for i, band in enumerate(bands):
                ds.write(a[i, ...], indexes=int(band + 1), window=window)

    @staticmethod
    def parse_slice(s: Union[slice, int, tuple, None], shape: Tuple[int]) -> tuple:
        """Parse a variable-format slice into predictable raster-based parameters

        Args:
            s (Union[slice, int, tuple, None]): Slice parameter
            shape: (Tuple[int]): Shape of the raster being sliced. Must be 3 dims.

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
        bands = get_slice_item(s[0], shape[0], True)

        # Collect remaining two dimensions
        row_off, height = get_slice_item(s[1], shape[1])
        col_off, width = get_slice_item(s[2], shape[2])

        if max(bands) > shape[0]:
            raise IndexError(
                f"Band index {max(bands)} out of bounds with band count {shape[0]}"
            )
        if row_off > shape[1] - 1:
            raise IndexError(
                f"Index {row_off} out of bounds for axis 0 with shape {shape[1]}"
            )
        if col_off > shape[2] - 1:
            raise IndexError(
                f"Index {col_off} out of bounds for axis 1 with shape {shape[2]}"
            )

        return bands, col_off, row_off, width, height

    def slice_params(self, s: Union[slice, int, tuple, None]) -> tuple:
        """Prepare a slice argument

        Args:
            s (Union[slice, int, tuple, None]): Slice parameter

        Returns:
            (tuple): bands, col_off, row_off, width, height
        """
        return self.parse_slice(s, self.shape)

    @contextmanager
    def ds(self, mode="r") -> Generator[rasterio.DatasetReader, None, None]:
        """Open the underlying raster dataset

        Returns:
            (rasterio.DatasetReader): Dataset for array collection
        """
        ds = rasterio.open(self.src, mode)
        yield ds
        ds.close()

    def matches(
        self, other_raster: Union[str, "Raster"], tolerance: float = 1e-6
    ) -> bool:
        """Check if the spatial parameters matche another raster

        Args:
            other_raster (Union[str, Raster]): Raster to compare against
        Returns:
            (bool): True if the rasters match, False otherwise
        """
        if isinstance(other_raster, str):
            other_raster = Raster(other_raster)

        return (
            self.shape == other_raster.shape
            and self.wkt == other_raster.wkt
            and np.isclose(self.csx, other_raster.csx, atol=tolerance)
            and np.isclose(self.csy, other_raster.csy, atol=tolerance)
            and np.isclose(self.top, other_raster.top, atol=tolerance)
            and np.isclose(self.left, other_raster.left, atol=tolerance)
            and np.isclose(self.bottom, other_raster.bottom, atol=tolerance)
            and np.isclose(self.right, other_raster.right, atol=tolerance)
        )

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure

        figure(figsize=(8, 6), dpi=80)

        a = np.ma.masked_equal(np.squeeze(self[:]), self.nodata)

        plt.imshow(a)
        plt.show()

    def data_and_index_iter(
        self, band=1
    ) -> Generator[Tuple[list, List[List[int]]], None, None]:
        """Iterate values and index locations from valid data

        Args:
            band (int, optional): Band to collect data from. Defaults to 1.

        Returns:
            (Generator[Tuple[list, List[List[int]]], None, None]): Values and a list
            of indices.
        """
        with rasterio.open(self.src) as rast:
            for _, window in rast.block_windows(band):
                a = rast.read(band, window=window)

                i, j = np.where(a != self.nodata)

                yield a[(i, j)].tolist(), [
                    [i_, j_] for i_, j_ in zip(i + window.row_off, j + window.col_off)
                ]

    def data_and_index(self, band: int = 1) -> Tuple[list, List[List[int]]]:
        """Collect values and index locations from valid data

        Args:
            band (int, optional): Band to collect data from. Defaults to 1.

        Returns:
            (Tuple[list, List[List[int]]]): Values and index locations in the form
            [[i, j], ...]
        """
        values = []
        indices = []

        for v, i in self.data_and_index_iter(band):
            values += v
            indices += i

        return values, indices


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


def to_raster(
    data: da.Array, template: str, destination: str, as_cog: bool = True, **kwargs
):
    """Compute a dask array and store it in a destination raster dataset

    Args:
        data (da.Array): Array of data to compute and store in the output raster
        template (str): Raster to use for specifications
        destination (str): Output raster path.
        as_cog (bool): The output raster will be a valid COG. Otherwise, it will be a
        compressed GeoTiff without overviews.
    """
    # The nodata value should be inferred from the data type and filled
    dtype = data.dtype.name

    if dtype == "bool":
        dtype = "uint8"
        data = da.ma.masked_where(~data, data.astype("uint8"))

    nodata = kwargs.get("nodata_value", infer_nodata(data))

    da.ma.set_fill_value(data, nodata)

    if data.ndim == 1:
        data = data.reshape((1, 1) + data.shape)
    elif data.ndim == 2:
        data = data.reshape((1,) + data.shape)

    def save_tif(data, src, dst):
        da.store(
            [da.ma.filled(data)],
            [
                Raster.empty_like(
                    dst,
                    src,
                    dtype=dtype,
                    nodata=nodata,
                    shape=data.shape,
                )
            ],
        )

    with ProgressBar():
        if as_cog:
            with TempRasterFile() as tmp_dst:
                save_tif(data, template, tmp_dst)
                translate_to_cog(tmp_dst, destination)

        else:
            save_tif(data, template, destination)
            try:
                run(["gdal_edit.py", "-stats", destination], check=True)
            except:
                pass


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


def reset_corrupt_blocks(src: str, dst: str):
    """Traverse a raster and replace corrupt blocks

    Args:
        src (str): Input raster
        dst (str): Output raster
    """
    out_rast = Raster.empty_like(dst, src)
    nodata = out_rast.nodata
    dtype = out_rast.dtype

    unreadable_blocks = 0
    total_blocks = 0

    with rasterio.open(dst, "r+") as dst_ds:
        with rasterio.open(src) as src_ds:
            for band in range(1, src_ds.count + 1):
                for _, window in src_ds.block_windows(band):
                    total_blocks += 1
                    try:
                        a = src_ds.read(band, window=window)
                    except:
                        unreadable_blocks += 1
                        a = np.full((window.height, window.width), nodata, dtype)

                    dst_ds.write(a, indexes=band, window=window)

    if unreadable_blocks > 0:
        print(f"Filled {unreadable_blocks} of {total_blocks} blocks with no data")


def clip_raster(
    src: str,
    mask: str,
    dst: str,
    crop_to_data: bool = False,
    resample_method: str = "bilinear",
):
    """Clip a source raster to the extent of another raster

    Args:
        src (string): Source raster dataset to be clipped
        mask (string): Raster used as a mask for clipping
        dst (string): Output (clipped) raster dataset
        crop_to_data (boolean): Reduce the extent of the output raster to where valid
        data exists. Defaults to False.
        resample_method (string): GDAL-supported resampling interpolation method. Defaults to 'bilinear'.
    """
    src_ds = Raster(src)
    mask_ds = Raster(mask)

    if crop_to_data:
        # Collect an extent from valid data in the mask
        top, bottom, right, left = -np.inf, np.inf, -np.inf, np.inf
        with mask_ds.ds() as ds:
            for band in range(1, ds.count + 1):
                for _, window in ds.block_windows(band):
                    a = ds.read(band, window=window)
                    i, j = np.where(a != mask_ds.nodata)

                    if i.size > 0:
                        top = max(
                            (mask_ds.top - window.row_off * mask_ds.csy)
                            - (i.min() * mask_ds.csy),
                            top,
                        )
                        bottom = min(
                            (
                                (
                                    mask_ds.top
                                    - (window.row_off + window.height) * mask_ds.csy
                                )
                                - (i.max() * mask_ds.csy)
                                - mask_ds.csy
                            ),
                            bottom,
                        )
                        left = min(
                            (mask_ds.left + window.col_off * mask_ds.csx)
                            + (j.min() * mask_ds.csx),
                            left,
                        )
                        right = max(
                            (
                                (
                                    mask_ds.left
                                    + (window.col_off + window.width) * mask_ds.csx
                                )
                                + (j.max() * mask_ds.csx)
                                + mask_ds.csx
                            ),
                            right,
                        )

    else:
        top, bottom, right, left = (
            mask_ds.top,
            mask_ds.bottom,
            mask_ds.right,
            mask_ds.left,
        )

    # Translate using the new extent
    with TempRasterFile() as mask_dst:
        if crop_to_data:
            cmd = [
                "gdal_translate",
                "-projwin",
                str(left),
                str(top),
                str(right),
                str(bottom),
                mask,
                mask_dst,
            ]
            run(cmd, check=True)

        else:
            mask_dst = mask

        with TempRasterFile() as tmp_dst:
            warp_like(
                src,
                tmp_dst,
                mask_dst,
                dtype=src_ds.dtype,
                resample_method=resample_method,
            )

            to_raster(
                da.ma.masked_where(
                    da.ma.getmaskarray(from_raster(mask_dst)), from_raster(tmp_dst)
                ),
                mask_dst,
                dst,
            )


def xy_align(
    src_1: str,
    src_2: str,
    dst_1: str,
    dst_2: str,
    align: str = "min",
    resample_method: str = "bilinear",
):
    """Align two rasters by the intersection of their extents and resolution

    Args:
        src_1 (str): Input Raster 1
        src_2 (str): Input Raster 2
        dst_1 (str): Output Raster 1
        dst_2 (str): Output Raster 2
        align (str, optional): Align with the minimum resolution using "min" or maximum resolution using "max".
        resample_method (str, optional): _description_. Defaults to "bilinear".
    """
    r1 = Raster(src_1)
    r2 = Raster(src_2)

    top = max(r1.top, r2.top)
    bottom = min(r1.bottom, r2.bottom)
    left = min(r1.left, r2.left)
    right = max(r1.right, r2.right)

    cs_method = min if align.lower() == "min" else max

    csx = cs_method(r1.csx, r2.csx)
    csy = cs_method(r1.csy, r2.csy)

    warp(
        src_1,
        dst_1,
        (left, bottom, right, top),
        csx,
        csy,
        resample_method=resample_method,
    )

    warp(
        src_2,
        dst_2,
        (left, bottom, right, top),
        csx,
        csy,
        resample_method=resample_method,
    )
