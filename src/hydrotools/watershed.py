from collections import OrderedDict
from copy import deepcopy
from logging import warning
import math
import os
import pickle
import gzip
from tempfile import TemporaryDirectory
from typing import Union

import dask.array as da
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin, rowcol
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from numba import njit
import fiona
from fiona.crs import from_string
from pyproj import CRS
import shapely
from shapely.geometry import mapping, Point
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from hydrotools.config import TMP_DIR
from hydrotools.utils import (
    GrassRunner,
    infer_nodata,
    translate_to_cog,
    indices_to_coords,
    transform_points,
    proj4_string,
)
from hydrotools.raster import (
    Raster,
    from_raster,
    to_raster,
    TempRasterFile,
    vectorize,
    warp,
    warp_like,
)
from hydrotools.data import (
    MRDEM_DTM,
    REMOTE_WARP_ARGS,
    collect_elevation,
    native_cell_size,
    transform_geom,
    upstream_basins,
    valid_footprint,
)


def flow_direction_accumulation(
    dem: str,
    direction_grid: str,
    accumulation_grid: str,
    single: bool = True,
    positive_only: bool = True,
    memory: Union[int, None] = None,
    beautify: Union[bool, None] = None,
):
    """Generate Flow Direction and Flow Accumulation grids using GRASS r.watershed

    Args:
        dem (str): Digital Elevation Model raster
        direction_grid (str): Output flow direction raster
        accumulation_grid (str): Output flow accumulatiorn raster
        single (bool, optional): Output Single Flow Direction. Defaults to True.
        positive_only (bool, optional): Only include positive flow direction values.
        Defaults to True.
        memory (Union[int, None], optional): Manage memory during execution by assigning
        a maximum block size. Defaults to 4096 MB.
        beautify (Union[bool, None]): make streams follow nicer paths over flat regions.
    """
    flags = ""
    if positive_only:
        flags += "a"
    if single:
        flags += "s"
    if memory is not None:
        flags += "m"
    if beautify:
        flags += "b"

    kwargs = {"memory": memory} if memory is not None else {}

    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.watershed",
            (dem, "dem", "raster"),
            elevation="dem",
            drainage="fd",
            accumulation="fa",
            flags=flags,
            **kwargs,
        )
        gr.save_raster("fd", direction_grid)
        gr.save_raster("fa", accumulation_grid)


def area_to_cells(src: str, area: float):
    """Convert an area into a number of raster cells

    Args:
        src (str): Input raster dataset
        area (float): An area in the raster's spatial reference
    """
    r = Raster(src)
    return int(np.ceil(area / (r.csx * r.csy)))


# Conservative estimate of WhiteboxTools memory use per DEM cell during
# BreachDepressionsLeastCost (input, output, and working grids)
BREACH_BYTES_PER_CELL = 40

# Attempts allowed per chunk before failing
BREACH_ATTEMPTS = 5


def breach_depressions(
    dem: str,
    dst: str,
    distance: float,
    max_memory: int = 4096,
    fill: bool = True,
):
    """Breach depressions in a DEM using the WhiteboxTools BreachDepressionsLeastCost tool

    The DEM is processed in chunks when it is estimated to exceed the maximum memory.
    Chunks are written to compressed temporary files with an overlap equal to the
    breach distance, processed sequentially, and the chunk interiors are
    reassembled into the output.

    Note, breaching is local to the breach distance, so chunked results are nearly
    identical to a single pass. Filling remaining depressions (`fill=True`) is not
    local, however, so filled areas may differ from a single pass where chunking
    applies (WhiteboxTools treats chunk edges as outlets). Use `fill=False` to
    produce results that are independent of chunking.

    This requires the optional `whitebox` package (`pip install whitebox`).

    Args:
        dem (str): Single-band Digital Elevation Model raster.
        dst (str): Output breached DEM path (a COG).
        distance (float): Maximum breach channel length in the DEM's spatial units.
        Converted to cells using the smaller cell dimension, rounding up.
        max_memory (int, optional): Maximum memory (MB) to use for processing a single
        chunk. Defaults to 4096 MB.
        fill (bool, optional): Fill depressions that remain after breaching.
        Defaults to True.
    """
    try:
        from whitebox import WhiteboxTools
    except ImportError as e:
        raise ImportError(
            "breach_depressions requires the optional dependency whitebox: "
            "pip install whitebox"
        ) from e

    if distance <= 0:
        raise ValueError("The breach distance must be greater than zero")

    r = Raster(dem)

    if r.band_count != 1:
        raise ValueError("The DEM must have a single band")

    nodata = r.nodata if r.nodata is not None else infer_nodata(r.dtype)
    _, height, width = r.shape

    dist_cells = int(math.ceil(distance / min(abs(r.csx), abs(r.csy))))

    # Determine the size of a chunk (including overlap) that fits in memory
    max_cells = int(max_memory * 1024**2 // BREACH_BYTES_PER_CELL)
    side = int(math.sqrt(max_cells))
    core_side = side - 2 * dist_cells
    if core_side < 1:
        raise ValueError(
            f"{max_memory} MB is too little memory to process a breach distance "
            f"of {dist_cells} cells. Increase max_memory or reduce distance."
        )
    core_rows = min(core_side, height)
    core_cols = min(core_side, width)

    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_compress_rasters(True)

    with TemporaryDirectory(dir=TMP_DIR) as tmp_dir:
        assembled = os.path.join(tmp_dir, "assembled.tif")
        Raster.empty_like(assembled, dem, nodata=nodata)

        with rasterio.open(dem) as src, rasterio.open(assembled, "r+") as out:
            for i, row in enumerate(range(0, height, core_rows)):
                for j, col in enumerate(range(0, width, core_cols)):
                    core = Window(
                        col,
                        row,
                        min(core_cols, width - col),
                        min(core_rows, height - row),
                    )

                    # Expand the core by the breach distance, clipped to the DEM
                    r0 = max(row - dist_cells, 0)
                    c0 = max(col - dist_cells, 0)
                    r1 = min(row + core.height + dist_cells, height)
                    c1 = min(col + core.width + dist_cells, width)
                    window = Window(c0, r0, c1 - c0, r1 - r0)

                    chunk_src = os.path.join(tmp_dir, f"chunk_{i}_{j}.tif")
                    chunk_dst = os.path.join(tmp_dir, f"chunk_{i}_{j}_breached.tif")

                    profile = src.profile.copy()
                    profile.update(
                        driver="GTiff",
                        height=window.height,
                        width=window.width,
                        transform=src.window_transform(window),
                        nodata=nodata,
                        tiled=True,
                        blockxsize=512,
                        blockysize=512,
                        compress="LZW",
                        bigtiff="YES",
                    )
                    with rasterio.open(chunk_src, "w", **profile) as chunk:
                        chunk.write(src.read(1, window=window), 1)

                    # WhiteboxTools intermittently panics when finalizing its
                    # output (a thread race), so retry before failing
                    for _ in range(BREACH_ATTEMPTS):
                        wbt.breach_depressions_least_cost(
                            chunk_src, chunk_dst, dist_cells, fill=fill
                        )
                        if os.path.isfile(chunk_dst):
                            break
                    else:
                        raise RuntimeError(
                            f"BreachDepressionsLeastCost failed for chunk ({i}, {j})"
                        )

                    with rasterio.open(chunk_dst) as breached:
                        data = breached.read(1)
                        chunk_nodata = breached.nodata

                    if chunk_nodata is not None and chunk_nodata != nodata:
                        data = np.where(
                            np.isclose(data, chunk_nodata), nodata, data
                        )

                    # Write only the core of the chunk
                    r_off, c_off = row - r0, col - c0
                    out.write(
                        data[
                            r_off : r_off + core.height, c_off : c_off + core.width
                        ].astype(r.dtype),
                        1,
                        window=core,
                    )

                    os.remove(chunk_src)
                    os.remove(chunk_dst)

        translate_to_cog(assembled, dst)


def auto_basin(
    dem: str, min_area: float, basin_dataset: str, memory: Union[int, None] = 4096
):
    """Automatically generate basins throughout a dataset that are larger than a minimum area

    Args:
        dem (str): Digital Elevation Model raster.
        min_area (float): Minimum basin area to constrain size.
        basin_dataset (str): Output raster data containing labeled basins.
        memory (Union[int, None], optional): Manage memory during execution by assigning
        a maximum block size. Defaults to 4096 MB.
    """
    # Min Area needs to be converted to cells
    min_area_cells = area_to_cells(dem, min_area)

    flags = "s"

    if memory is not None:
        flags += "m"

    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.watershed",
            (dem, "dem", "raster"),
            elevation="dem",
            threshold=min_area_cells,
            basin="b",
            flags=flags,
            memory=memory if memory is not None else 300,  # r.watershed default
        )
        gr.save_raster("b", basin_dataset)


def extract_streams(
    dem: str,
    accumulation_src: str,
    stream_dst: str,
    direction_dst: str,
    min_area: float = 1e6,
    min_length: float = 0,
    mexp: float = None,
    memory: Union[int, None] = 4096,
):
    """Extract simulated streams

    Args:
        dem (str): Path to a Digital Elevation Model (DEM)
        accumulation_src (str): Flow Accumulation dataset derived from
        `flow_direction_accumulation`
        stream_dst (str): Output Streams raster path
        direction_dst (str): Output flow direction raster path
        min_area (float, optional): Minimum watershed area for streams. Defaults to 1e6.
        min_length (float, optional): Minimum stream length. Defaults to 0.
        memory (Union[int, None], optional): Manage memory during execution by assigning
        a maximum block size. Defaults to 4096 MB.
    """
    # Min Area needs to be converted to cells
    min_area_cells = area_to_cells(dem, min_area)

    kwargs = {
        "elevation": "dem",
        "accumulation": "accum",
        "stream_length": float(min_length),
        "threshold": min_area_cells,
        "stream_raster": "streams",
        "direction": "direction",
        "memory": memory,
    }

    if mexp is not None:
        kwargs["mexp"] = mexp

    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.stream.extract",
            (dem, "dem", "raster"),
            (accumulation_src, "accum", "raster"),
            **kwargs,
        )
        gr.save_raster("streams", stream_dst)
        gr.save_raster("direction", direction_dst)


def stream_basins(
    fd: str,
    stream_src: str,
    basins_dst: str,
    smooth_corners: bool = True,
    memory: Union[int, None] = 4096,
):
    """Create basins for a stream network extracted using `extract_streams`

    Args:
        fd (str): An input Flow Direction grid.
        stream_src (str): Streams derived using `extract_streams`.
        basins_dst (str): An output vector file for basins.
        memory (Union[int, None], optional): Manage memory during execution by assigning
        a maximum block size. Defaults to 4096 MB.
    """
    flags = ""
    if memory is not None:
        flags += "m"

    kwargs = {"memory": memory} if memory is not None else {}

    with TempRasterFile() as basins_rast:
        with GrassRunner(fd) as gr:
            gr.run_command(
                "r.stream.basins",
                (fd, "fd", "raster"),
                (stream_src, "streams", "raster"),
                direction="fd",
                stream_rast="streams",
                basins="basins",
                **kwargs,
            )
            gr.save_raster("basins", basins_rast)

        vectorize(basins_rast, basins_dst, smooth_corners=smooth_corners)


def stream_order(
    dem: str,
    stream_src: str,
    direction_src: str,
    accumulation_src: str,
    order_dst: str,
    use_accum: bool = False,
    zero_bg: bool = False,
    memory: Union[int, None] = 4096,
):
    """Calculate stream order using the following data:

    * Streams and flow direction derived from `hydrotools.watershed.extract_streams`

        ::Note:: The flow direction from
        `hydrotools.watershed.flow_direction_accumulation` may not be used in
        conjunction with streams derived from `hydrotools.watershed.extract_streams`!

    * Flow Accumulation derived from `hydrotools.watershed.flow_direction_accumulation`

    Args:
        dem (str): An input Digital Elevation Model.
        stream_src (str): Streams derived using `extract_streams`.
        direction_src (str): Flow Direction derived from `extract_streams`.
        accumulation_src (str): Flow Accumulation derived from
        `flow_direction_accumulation`.
        order_dst (str): An output stream order dataset. If the extension `.tif` is
        used, a Strahler stream order is output, otherwise a geopackage vector is
        created.
        use_accum (bool): Use flow accumulation to trace Horton and Hack orders.
        zero_bg (bool): Use a background value of 0 instead of nodata
        memory (Union[int, None], optional): Manage memory during execution by assigning
        a maximum block size. Defaults to 4096 MB.
    """
    if order_dst.lower().endswith(".tif"):
        kwargs = {"strahler": "stream_o"}
    else:
        kwargs = {"stream_vect": "stream_o"}

    if memory is not None:
        kwargs["memory"] = memory

    flags = ""
    if zero_bg:
        flags += "z"
    if memory is not None:
        flags += "m"
    if use_accum:
        flags += "a"

    with GrassRunner(dem) as gr:
        # r.stream.topology is the r.stream.order tool with the stream order portions removed
        gr.run_command(
            "r.stream.order",
            (dem, "dem", "raster"),
            (stream_src, "streams", "raster"),
            (direction_src, "direction", "raster"),
            (accumulation_src, "accum", "raster"),
            elevation="dem",
            stream_rast="streams",
            direction="direction",
            accumulation="accum",
            flags=flags,
            **kwargs,
        )

        if order_dst.lower().endswith(".tif"):
            gr.save_raster("stream_o", order_dst)
        else:
            if not order_dst.lower().endswith(".gpkg"):
                order_dst += ".gpkg"

            gr.run_command(
                "v.extract",
                input="stream_o",
                output="stream_order",
                type="line",
            )

            gr.save_vector("stream_order", order_dst)


def basin(flow_direction: str, x: float, y: float, dst: str):
    """Delineate a watershed from coordinates using `r.water.outlet`

    Args:
        flow_direction (str): Path to a flow direction grid
        x (float): Coordinate in the x-direction
        y (float): Coordinate in the y-direction
        dst (str): Output path for the watershed vector file
    """
    raster_type = dst.lower().endswith(".tif")

    with TempRasterFile() as tmp_rast:
        with GrassRunner(flow_direction) as gr:
            gr.run_command(
                "r.water.outlet",
                (flow_direction, "fd", "raster"),
                input="fd",
                output="basin",
                coordinates=(x, y),
            )

            gr.save_raster(
                "basin", dst if raster_type else tmp_rast, as_cog=raster_type
            )

        if not raster_type:
            vectorize(tmp_rast, dst, smooth_corners=True)


class FlowAccumulation:
    """
    Use the GRASS method r.flowaccumulation to summarize sums
    and means of raster parameters.

    NOTE: This requires the r.flowaccumulation GRASS addon
    """

    def __init__(self, direction: str):
        """Initialize a FlowAccumulation analysis

        Args:
            direction (str): Path to a flow direction raster generated using r.stream.extract
        """
        self.direction = direction

        self.wkdir = TemporaryDirectory()

        self._modals = None

    @property
    def modals(self):
        if self._modals is None:
            self._modals = os.path.join(self.wkdir.name, "modals.tif")
            with GrassRunner(self.direction) as gr:
                gr.run_command(
                    "r.flowaccumulation",
                    (self.direction, "direction", "raster"),
                    direction="direction",
                    output="modals",
                    type="FCELL",
                )
                gr.save_raster("modals", self._modals, as_cog=False)

        return self._modals

    def calculate(self, raster_source: str, dst: str, method: str = "sum"):
        """Calculate flow accumulation statistics

        Args:
            raster_source (str): Path to a raster dataset to calculate stats on
            dst (str): Path to save the output geopackage
        """
        # Ensure the raster_source aligns with the input grids
        if not Raster(raster_source).matches(self.direction):
            raise ValueError("Input raster does not align with direction grid")

        # Accumulate
        sum_result = (
            dst if method == "sum" else os.path.join(self.wkdir.name, "sum_result.tif")
        )
        with GrassRunner(self.direction) as gr:
            gr.run_command(
                "r.flowaccumulation",
                (self.direction, "direction", "raster"),
                (raster_source, "source", "raster"),
                direction="direction",
                weight="source",
                output="sum_result",
                type="DCELL",
            )
            gr.save_raster("sum_result", sum_result, as_cog=True)

        if method == "mean":
            modals = from_raster(self.modals)
            avg = da.ma.masked_where(
                modals == 0, from_raster(sum_result) / modals
            ).astype("float32")

            to_raster(avg, self.direction, dst)

    def contributing_area(self, dst: str):
        """Save the general contributing area raster in km**2

        Args:
            dst (str): Path to the contributing area raster
        """
        r = Raster(self.modals)
        ca = (from_raster(self.modals) * r.csx * r.csy) / 1e6

        to_raster(
            da.ma.masked_where(ca == 0, ca),
            self.direction,
            dst,
        )

    def __del__(self):
        """Clean up temporary working directory"""
        self.wkdir.cleanup()


class WatershedIndex:
    """
    __DEPRECATED__: Use `hydrotools.watershed.FlowAccumulation` instead

    Build and cache an index of all contributing cells to every location on  a stream grid

    To create a new index, save to a file, and initialize the index:
    ```
    wi = WatershedIndex.create_index(
        "flow_direction.tif", "flow_accumulation.tif", "index_file"
    )
    ```

    To initialize with an index file:
    ```
    wi = WatershedIndex("index_file")
    ```

    To run stats on a dataset:
    ```
    wi.calculate_stats("a_dataset.tif", "destination_ds")
    ```

    Returned stats include "min", "max", "sum", and "mean"
    """

    def __init__(self, idx_path: str):
        """Load an existing index from a path

        Args:
            idx_path (str): Path to previously created index
            idx (list): Existing index, which avoids the need to re-read the index from
            disk. This argument is reserved for the `create_index` classmethod and
            should not be used directly without caution.
        """
        self.path = idx_path

        with gzip.open(idx_path, "rb") as f:
            # Initialize from the header
            self.__dict__.update(pickle.load(f))

        self.inmem_idx = None

    @property
    def idx(self) -> list:
        """Load the entire index from the initialization file into memory

        Returns:
            list: Watershed index graph of nested lists
        """
        if self.inmem_idx is not None:
            return self.inmem_idx

        with gzip.open(self.path, "rb") as f:
            # Seek past the header
            _ = pickle.load(f)
            self.inmem_idx = []

            while True:
                try:
                    self.inmem_idx.append(pickle.load(f))
                except EOFError:
                    break

        return self.inmem_idx

    def __iter__(self):
        """Iterate over each watershed in the index

        Yields:
            tuple: A tuple of (contributing_index, nested_index) for each watershed
        """
        with gzip.open(self.path, "rb") as f:
            # Seek past the header
            _ = pickle.load(f)

            while True:
                try:
                    ws = pickle.load(f)
                except EOFError:
                    break

                yield ws

    @classmethod
    def index_from_dem(
        cls, dem: str, index_path: str, minimum_area: float = 1e6, **kwargs
    ):
        """Compute flow direction and flow accumulation prior to creating a new index

        Args:
            dem (str): Path to an input Digital Elevation Model (DEM)
            index_path: Path to save the Watershed Index
            minimum_area (float, optional): Minimum contributing area used to classify
            streams. Defaults to 1e6.

        Kwargs:
            Those passed to `flow_direction_accumulation`

        Raises:
            ValueError: _description_
            e: _description_

        Returns:
            WatershedIndex: Instance of self
        """
        with TemporaryDirectory() as tmp_dir:
            direction_grid = os.path.join(tmp_dir, "direction.tif")
            accumulation_grid = os.path.join(tmp_dir, "accumulation.tif")

            flow_direction_accumulation(
                dem, direction_grid, accumulation_grid, **kwargs
            )

            return cls.create_index(
                direction_grid, accumulation_grid, index_path, minimum_area
            )

    @classmethod
    def create_index(
        cls,
        direction_raster: str,
        stream_raster: str,
        index_path: str,
    ):
        """Create a new watershed index

        All watersheds are stored as gzipped and pickled objects, which include:
            _watersheds_
            Lists of contributing cells to a point on a stream in the form:
            `contributing_index = [ stream point 0: [[i1, j1], [i2, j2]...[in, jn]],...stream point n: [[]] ]`
            where i and j are coordinates of contributing cells.
            _Nesting of watersheds_
            A list of the watershed nesting hierarchy in the form:
            `nested_index = [ stream point 0: [i1, i2, i3...in],...stream point n: []]`
            where i is the index of the stream point that falls within the stream point index

        Args:
            direction_raster (str): Flow direction raster generated using
            `hydrotools.watershed.flow_direction_accumulation`. Note: the `single`
            argument must be set to True
            stream_raster (str): Raster with streams using
            `hydrotools.watershed.extract_streams`.
            index_path (str): Path to create a new index

        Returns:
            WatershedIndex: Instance of self
        """
        domain_spec = Raster.raster_specs(direction_raster)

        # Load into memory
        fd = np.squeeze(from_raster(direction_raster).compute())

        # Create a boolean mask of streams
        streams = np.squeeze(~da.ma.getmaskarray(from_raster(stream_raster)).compute())

        # Initialize an array for the stack
        visited = np.ma.getmaskarray(fd)

        fd = fd.filled(-1)

        @njit
        def traverse(fd, streams, i, j):
            directions = [
                None,
                [-1, 1],
                [-1, 0],
                [-1, -1],
                [0, -1],
                [1, -1],
                [1, 0],
                [1, 1],
                [0, 1],
            ]

            while True:
                # Off map
                if fd[i, j] <= 0:
                    break

                # Collect the downstream cell
                i_offset, j_offset = directions[fd[i, j]]
                new_i, new_j = i + i_offset, j + j_offset

                # If the dirs are only positive, end when not on a stream
                if not streams[new_i, new_j]:
                    break

                i, j = new_i, new_j

            return i, j

        def next_ws():
            candidates = streams & ~visited
            if candidates.sum() == 0:
                return

            i, j = np.unravel_index(np.argmax(candidates), streams.shape)

            # Traverse to the outlet
            return traverse(fd, streams, i, j)

        @njit
        def delineate(fd, streams, i, j, visited):
            # i: 22980, j: 16240
            directions = [[7, 6, 5], [8, 0, 4], [1, 2, 3]]

            # Elements contributing to each coordinate - the first element is always on the stream
            ci = [[(i, j)]]
            # Used to track elements still requiring evaluation of neighbours
            ci_e = [0]
            # Contributing indexes to ni. Use -1 to initiate the list with a type
            ni = [[-1]]

            # Mark the seed element as visited
            visited[i, j] = True

            cursor = 0
            while True:
                # print("Collect a new element to test")
                if ci_e[cursor] < len(ci[cursor]):
                    i, j = ci[cursor][ci_e[cursor]]
                    ci_e[cursor] += 1
                else:
                    # print("Backtrack or break out of the algo")
                    cursor -= 1
                    if cursor < 0:
                        break
                    continue

                # print("Test the current element at location (i, j)")
                stream_elems = []
                for row_offset in range(-1, 2):
                    for col_offset in range(-1, 2):
                        t_i, t_j = i + row_offset, j + col_offset

                        if (
                            t_i < 0
                            or t_i >= fd.shape[0]
                            or t_j < 0
                            or t_j >= fd.shape[1]
                        ):
                            continue

                        if visited[t_i, t_j]:
                            continue

                        # print("Check if the element at this offset contributes to the element being tested")
                        if fd[t_i, t_j] == directions[row_offset + 1][col_offset + 1]:
                            # print("This element has now been visited")
                            visited[t_i, t_j] = True

                            if streams[t_i, t_j]:
                                # print("This element comprises a stream - add as a nested element")
                                stream_elems.append((t_i, t_j))
                            else:
                                # print("Add to contributing stack, and the testing queue")
                                ci[cursor].append((t_i, t_j))

                # print(" Add nested locations and propagate past any stream elements")
                this_index = cursor
                for se in stream_elems:
                    # print("Add nested to current")
                    cursor = len(ci_e)
                    ni[this_index].append(cursor)
                    # print("New list item")
                    ci.append([se])
                    ci_e.append(0)
                    ni.append([-1])

            return ci, [[j for j in i if j != -1] for i in ni]

        with gzip.open(index_path, "wb") as f:
            pickle.dump(domain_spec, f)

            # Run the alg
            coord = next_ws()
            watershed_cnt = 0
            while coord is not None:
                watershed_cnt += 1
                i, j = coord

                ci, ni = delineate(fd, streams, i, j, visited)

                pickle.dump((ci, ni), f)

                coord = next_ws()

        print(f"Index at {index_path} created")

        return cls(index_path)

    def save_locations(
        self, dst: str, data: dict = None, output_crs: Union[str, int] = None
    ):
        """Save the underlying locations of the index to a vector file

        Args:
            dst (str): Path for the output geopackage
        """
        if not dst.lower().endswith("gpkg"):
            dst += ".gpkg"

        points = []
        networks = []
        network_id = 0
        for coords, _ in self:
            networks += [network_id] * len(coords)
            network_id += 1
            y, x = indices_to_coords(
                (
                    [coord[0][0] for coord in coords],
                    [coord[0][1] for coord in coords],
                ),
                self.top,
                self.left,
                self.csx,
                self.csy,
            )

            points += list(zip(x, y))

        if output_crs is not None:
            points = transform_points(points, self.wkt, output_crs)

        # If data are not provided, this saves the base attributes
        if data is None:
            fields = [("networkid", "int")]
        else:
            fields = [(key, "float") for key in data.keys()]

        schema = {
            "geometry": "Point",
            "properties": OrderedDict([("id", "int")] + fields),
        }

        crs = (
            from_string(proj4_string(output_crs))
            if output_crs is not None
            else from_string(proj4_string(self.wkt))
        )

        records = [
            {
                "geometry": {"type": "Point", "coordinates": pnt},
                "properties": OrderedDict(
                    [("id", fid)]
                    + (
                        [("networkid", networks[fid])]
                        if data is None
                        else [(key, data[key][fid]) for key in data.keys()]
                    )
                ),
            }
            for fid, pnt in enumerate(points)
        ]

        with fiona.open(dst, "w", "GPKG", crs=crs, schema=schema) as layer:
            layer.writerecords(records)

    def calculate_stats(
        self,
        raster_source: str,
        destination: str,
        **kwargs,
    ):
        """Use a generated index to calculate stats at stream locations

        Args:
            raster_source (str): Raster dataset to collect stats from
            destination (str): Destination path to save the output geopackage

        Kwargs:
            output_crs (Union[int, str]): Output coordinate reference system for points.
            Defaults to None, whereby the domain CRS will be used.
        """
        print(f"Calculating stats for {os.path.basename(raster_source)}")

        src = Raster(raster_source)

        if any(
            [
                tuple(src.shape) != tuple(self.shape),
                not np.isclose(src.top, self.top),
                not np.isclose(src.left, self.left),
            ]
        ):
            raise ValueError("Input data must spatially match index domain")

        data = np.squeeze(from_raster(src).compute())
        m = (data != src.nodata) & ~np.isnan(data) & ~np.isinf(data)
        float_boundary = np.finfo("float32").max

        def add_stats(i, prv_i, elems, _min, _max, _sum, modals):
            elems = tuple(np.array(elems).T)
            sample = data[elems]
            sample = sample[m[elems]]

            if sample.size > 0:
                s_min = sample.min()
                _min[i] = min([_min[i], s_min])

                s_max = sample.max()
                _max[i] = max([_max[i], s_max])

                _sum[i] += sample.sum()
                modals[i] += sample.size

            if prv_i is not None:
                _min[prv_i] = min([_min[prv_i], _min[i]])
                _max[prv_i] = max([_max[prv_i], _max[i]])
                _sum[prv_i] += _sum[i]
                modals[prv_i] += modals[i]

        def summarize(args):
            ci, ni = args

            ni = deepcopy(ni)

            # Assign output datasets
            _min = np.zeros(len(ci), np.float32) + float_boundary
            _max = np.zeros(len(ci), np.float32) - float_boundary
            _sum = np.zeros(len(ci), np.float32)
            modals = np.zeros(len(ci), np.float32)

            stack = [0]
            cursor = 0
            while cursor is not None:
                try:
                    cursor = ni[cursor].pop()
                    stack.append(cursor)
                except IndexError:
                    del stack[-1]
                    next_cursor = stack[-1] if len(stack) > 0 else None
                    add_stats(cursor, next_cursor, ci[cursor], _min, _max, _sum, modals)
                    cursor = next_cursor

            nodata = modals == 0
            _mean = np.zeros(len(ci), np.float32) + float_boundary
            _mean[~nodata] = _sum[~nodata] / modals[~nodata]
            _max[_max == -float_boundary] *= -1
            _sum[nodata] = float_boundary

            return _min, _max, _sum, _mean

        stat_output = {
            "min": [],
            "max": [],
            "sum": [],
            "mean": [],
        }

        def map_results(res):
            return [
                val if val != np.finfo("float32").max else None for val in res.tolist()
            ]

        for ci, ni in self:
            _min, _max, _sum, _mean = summarize((ci, ni))
            stat_output["min"] += map_results(_min)
            stat_output["max"] += map_results(_max)
            stat_output["sum"] += map_results(_sum)
            stat_output["mean"] += map_results(_mean)

        print(f"Saving stat results to {os.path.basename(destination)}")

        self.save_locations(
            destination, data=stat_output, output_crs=kwargs.get("output_crs", None)
        )


# Row/col offsets indexed by GRASS flow direction (1-8 counterclockwise from NE, 0
# is unused)
FD_OFF_I = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int64)
FD_OFF_J = np.array([0, 1, 0, -1, -1, -1, 0, 1, 1], dtype=np.int64)

# Maximum number of cells used to screen the area draining into a DEM
SCREENING_CELLS = 4_000_000


@njit
def _drains_to(fd, valid, target, off_i, off_j):
    """Identify cells with a flow path that reaches a target region"""
    n, m = fd.shape

    # 0: unknown, 1: drains to target, 2: does not drain to target, 3: on current path
    state = np.zeros((n, m), np.uint8)
    for i in range(n):
        for j in range(m):
            if valid[i, j] and target[i, j]:
                state[i, j] = 1
            elif not valid[i, j]:
                state[i, j] = 2

    path_i = np.empty(n * m, np.int64)
    path_j = np.empty(n * m, np.int64)

    for i0 in range(n):
        for j0 in range(m):
            if state[i0, j0] != 0:
                continue

            length = 0
            i, j = i0, j0
            while True:
                s = state[i, j]
                if s == 3:
                    # Cycle
                    result = 2
                    break
                elif s != 0:
                    result = s
                    break

                path_i[length] = i
                path_j[length] = j
                length += 1
                state[i, j] = 3

                k = abs(fd[i, j])
                if k < 1 or k > 8:
                    result = 2
                    break

                i += off_i[k]
                j += off_j[k]
                if i < 0 or i >= n or j < 0 or j >= m:
                    result = 2
                    break

            for p in range(length):
                state[path_i[p], path_j[p]] = result

    return state == 1


@njit
def _accumulate(fd, valid, weights, off_i, off_j):
    """Accumulate any number of weights (stacked on the first axis) downstream"""
    n, m = fd.shape
    acc = weights.copy()

    indeg = np.zeros((n, m), np.int32)
    for i in range(n):
        for j in range(m):
            if not valid[i, j]:
                continue
            k = abs(fd[i, j])
            if k < 1 or k > 8:
                continue
            ii = i + off_i[k]
            jj = j + off_j[k]
            if 0 <= ii < n and 0 <= jj < m and valid[ii, jj]:
                indeg[ii, jj] += 1

    stack = np.empty(n * m, np.int64)
    top = 0
    for i in range(n):
        for j in range(m):
            if valid[i, j] and indeg[i, j] == 0:
                stack[top] = i * m + j
                top += 1

    while top > 0:
        top -= 1
        i = stack[top] // m
        j = stack[top] % m

        k = abs(fd[i, j])
        if k < 1 or k > 8:
            continue
        ii = i + off_i[k]
        jj = j + off_j[k]
        if 0 <= ii < n and 0 <= jj < m and valid[ii, jj]:
            for w in range(acc.shape[0]):
                acc[w, ii, jj] += acc[w, i, j]
            indeg[ii, jj] -= 1
            if indeg[ii, jj] == 0:
                stack[top] = ii * m + jj
                top += 1

    return acc


def _grid_bounds(bounds: tuple, cs: float) -> tuple:
    """Snap bounds outwards to a multiple of a cell size"""
    left, bottom, right, top = bounds

    return (
        math.floor(left / cs) * cs,
        math.floor(bottom / cs) * cs,
        math.ceil(right / cs) * cs,
        math.ceil(top / cs) * cs,
    )


def _regional_flow(
    sources: list,
    crs: str,
    bounds: tuple,
    cs: float,
    mask: np.ndarray,
    dem: str,
    wkdir: str,
    name: str,
) -> dict:
    """Warp regional DEM sources onto a grid, limit them to a mask, and calculate
    single flow directions using r.watershed.

    Returns:
        dict: Grid arrays and specifications
    """
    left, bottom, right, top = bounds
    shape_ = (int(round((top - bottom) / cs)), int(round((right - left) / cs)))
    if mask.shape != shape_:
        raise ValueError("The regional mask does not match the regional grid")

    regional_dem = os.path.join(wkdir, f"{name}_regional_dem.tif")
    masked_dem = os.path.join(wkdir, f"{name}_masked_dem.tif")
    target_path = os.path.join(wkdir, f"{name}_target.tif")
    fd_path = os.path.join(wkdir, f"{name}_fd.tif")

    collect_elevation(regional_dem, bounds, cs, crs, sources=sources, as_cog=False)

    with rasterio.open(regional_dem) as ds:
        a = ds.read(1)
        nodata = ds.nodata if ds.nodata is not None else infer_nodata(a)
        profile = ds.profile.copy()

    dem_valid = (a != nodata) & np.isfinite(a)
    a[~(dem_valid & mask)] = nodata
    profile.update(nodata=nodata)

    with rasterio.open(masked_dem, "w", **profile) as ds:
        ds.write(a, 1)

    del a

    # The footprint of the DEM, where any of its valid data falls in a regional cell
    warp_like(dem, target_path, masked_dem, resample_method="max", as_cog=False)
    with rasterio.open(target_path) as ds:
        target = ds.read_masks(1) > 0

    with GrassRunner(masked_dem) as gr:
        gr.run_command(
            "r.watershed",
            (masked_dem, "dem", "raster"),
            elevation="dem",
            drainage="fd",
            flags="s",
        )
        gr.save_raster("fd", fd_path, as_cog=False)

    with rasterio.open(fd_path) as ds:
        fd = ds.read(1).astype(np.int8)
        valid = (ds.read_masks(1) > 0) & dem_valid & mask

    return {
        "fd": fd,
        "valid": valid,
        "dem_valid": dem_valid,
        "target": target & valid,
        "bounds": bounds,
        "cs": cs,
        "transform": from_origin(left, top, cs, cs),
        "shape": shape_,
    }


@njit
def _walk_upstream(fd, valid, candidates, dist, i, j, off_i, off_j):
    """Walk upstream from a cell along the largest channel, returning the cell
    closest to the centre of the window (the inflow location)"""
    n, m = fd.shape
    weights = np.ones((1, n, m))
    acc = _accumulate(fd, valid, weights, off_i, off_j)[0]

    best_i, best_j = i, j
    while True:
        next_i, next_j = -1, -1
        next_acc = 0.0
        for k in range(1, 9):
            ii = i - off_i[k]
            jj = j - off_j[k]
            if (
                0 <= ii < n
                and 0 <= jj < m
                and candidates[ii, jj]
                and abs(fd[ii, jj]) == k
                and acc[ii, jj] > next_acc
            ):
                next_i, next_j = ii, jj
                next_acc = acc[ii, jj]

        if next_i < 0:
            break

        i, j = next_i, next_j
        if dist[i, j] < dist[best_i, best_j]:
            best_i, best_j = i, j

    return best_i, best_j


def _read_padded(ds, row0: int, col0: int, size: int, fill_value) -> np.ndarray:
    """Read a square window from the first band, filling areas outside the raster"""
    out = np.full((size, size), fill_value, dtype=ds.dtypes[0])

    r0, c0 = max(row0, 0), max(col0, 0)
    r1, c1 = min(row0 + size, ds.height), min(col0 + size, ds.width)
    if r1 > r0 and c1 > c0:
        out[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0] = ds.read(
            1, window=Window(c0, r0, c1 - c0, r1 - r0)
        )

    return out


def _snap_inflows(
    dem: str,
    direction: str,
    inflows: list,
    min_area: float,
    snap_distance: float,
) -> dict:
    """Snap inflows to interior DEM cells with a flow direction that remains in the
    DEM. Inflows larger than `min_area` are snapped to the receiving channel, found
    using the lowest cell within the snap distance and followed upstream towards the
    inflow location. Smaller inflows are snapped to the nearest cell.

    Returns:
        dict: Seeds in the form {(row, col): [area, precip * area, x, y, distance]}
    """
    r = Raster(dem)
    nodata = r.nodata
    rad = int(math.ceil(snap_distance / min(r.csx, r.csy)))
    size = 2 * rad + 3

    ii, jj = np.mgrid[0:size, 0:size]
    dist = np.sqrt(((ii - rad - 1) * r.csy) ** 2 + ((jj - rad - 1) * r.csx) ** 2)
    in_radius = dist <= snap_distance

    seeds = {}
    lost_area = 0.0

    with rasterio.open(dem) as dem_ds, rasterio.open(direction) as fd_ds:
        for inflow in inflows:
            row, col = rowcol(dem_ds.transform, inflow["x"], inflow["y"])
            row0, col0 = row - rad - 1, col - rad - 1

            z = _read_padded(dem_ds, row0, col0, size, nodata)
            fd = _read_padded(fd_ds, row0, col0, size, 0)

            valid = (z != nodata) & np.isfinite(z)
            # Cells surrounded by valid data, which flow to another valid cell
            candidates = (
                binary_erosion(valid, np.ones((3, 3), bool), border_value=0)
                & in_radius
                & (fd > 0)
            )

            if not candidates.any():
                lost_area += inflow["area"]
                continue

            if inflow["area"] >= min_area:
                ci, cj = np.unravel_index(
                    np.argmin(np.where(candidates, z, np.inf)), z.shape
                )
                ci, cj = _walk_upstream(
                    fd, valid, candidates, dist, ci, cj, FD_OFF_I, FD_OFF_J
                )
            else:
                ci, cj = np.unravel_index(
                    np.argmin(np.where(candidates, dist, np.inf)), z.shape
                )

            key = (int(row0 + ci), int(col0 + cj))

            seed = seeds.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            seed[0] += inflow["area"]
            seed[1] += inflow["area"] * inflow.get("precip", 0.0)
            if inflow["area"] >= seed[4]:
                x, y = dem_ds.xy(*key)
                seed[2], seed[3] = x, y

            seed[4] = max(seed[4], inflow["area"])

    if lost_area > 0:
        warning(
            f"{lost_area / 1e6:.3f} km2 of inflow could not be snapped to the DEM "
            "and was not included"
        )

    return seeds


def _add_seeds(block, rows, cols, values, block_info=None):
    """Add seed values to a dask block"""
    (_, _), (r0, r1), (c0, c1) = block_info[0]["array-location"]
    sel = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)

    if sel.any():
        block = block.copy()
        np.add.at(block[0], (rows[sel] - r0, cols[sel] - c0), values[sel])

    return block


def _seeded_weights(
    dem: str,
    template: str,
    seeds: dict,
    area_dst: str,
    precip: Union[str, None] = None,
    precip_dst: Union[str, None] = None,
):
    """Create cell area and (optionally) precip * cell area weights with added inflow
    seeds"""
    r = Raster(dem)
    cell_area = r.csx * r.csy

    dem_a = from_raster(dem)
    valid = ~da.ma.getmaskarray(dem_a)

    keys = list(seeds.keys())
    rows = np.array([k[0] for k in keys], dtype=np.int64)
    cols = np.array([k[1] for k in keys], dtype=np.int64)
    areas = np.array([seeds[k][0] for k in keys], dtype=np.float64)

    weights = [(da.where(valid, cell_area, np.nan).astype(np.float64), areas, area_dst)]

    if precip is not None:
        precip_a = from_raster(precip)
        missing = int((valid & da.ma.getmaskarray(precip_a)).sum().compute())
        if missing > 0:
            raise ValueError(
                f"The precipitation raster does not cover {missing} cells of the DEM"
            )

        precips = np.array([seeds[k][1] for k in keys], dtype=np.float64)
        precip_w = da.where(
            valid, da.ma.filled(precip_a, 0).astype(np.float64) * cell_area, np.nan
        )
        weights.append((precip_w, precips, precip_dst))

    for w, values, dst in weights:
        w = w.map_blocks(_add_seeds, rows, cols, values, dtype=np.float64)
        to_raster(da.ma.masked_invalid(w), template, dst, as_cog=False)


def _r_stream_extract(
    dem: str,
    stream_dst: str,
    direction_dst: str,
    threshold: int,
    min_length: float,
    accumulation: Union[str, None] = None,
    mexp: Union[float, None] = None,
    memory: Union[int, None] = 4096,
    as_cog: bool = True,
):
    """Run r.stream.extract with an optional accumulation input"""
    args = [(dem, "dem", "raster")]
    kwargs = {
        "elevation": "dem",
        "threshold": threshold,
        "stream_length": float(min_length),
        "stream_raster": "streams",
        "direction": "direction",
    }

    if accumulation is not None:
        args.append((accumulation, "accum", "raster"))
        kwargs["accumulation"] = "accum"

    if mexp is not None:
        kwargs["mexp"] = mexp

    if memory is not None:
        kwargs["memory"] = memory

    with GrassRunner(dem) as gr:
        gr.run_command("r.stream.extract", *args, **kwargs)
        gr.save_raster("streams", stream_dst, as_cog=as_cog)
        gr.save_raster("direction", direction_dst, as_cog=as_cog)


def extract_streams_with_inflows(
    dem: str,
    inflows: list,
    stream_dst: str,
    direction_dst: str,
    contributing_area_dst: str,
    precip: Union[str, None] = None,
    mean_precip_dst: Union[str, None] = None,
    min_area: float = 1e6,
    min_length: float = 0,
    mexp: Union[float, None] = None,
    snap_distance: float = 250.0,
    inflow_dst: Union[str, None] = None,
    memory: Union[int, None] = 4096,
):
    """Extract streams from a DEM that does not cover the full contributing area of
    its streams, by adding the area (and optionally precipitation) of known inflows.

    Inflows are snapped to the DEM (see `snap_distance`), and are used to seed flow
    accumulation of cell area and precipitation. Streams are then extracted using
    the corrected accumulation, and the contributing area and mean upstream
    precipitation are calculated on the resulting flow direction.

    Use `extract_streams_expanded` to calculate the inflows automatically.

    Args:
        dem (str): Digital Elevation Model raster in a projected (preferably
        equal-area) spatial reference with units of metres.
        inflows (list): Inflows in the form `{"x": float, "y": float, "area": float,
        "precip": float}`, where `x` and `y` are in the DEM spatial reference,
        `area` is the external contributing area (m2), and `precip` (only required
        with `precip`) is the mean precipitation over the external contributing area.
        stream_dst (str): Output streams raster.
        direction_dst (str): Output flow direction raster.
        contributing_area_dst (str): Output contributing area raster (km2).
        precip (Union[str, None], optional): Mean annual precipitation raster covering
        the DEM. It is warped to the DEM grid when it does not align. Defaults to None.
        mean_precip_dst (Union[str, None], optional): Output mean upstream
        precipitation raster. Required with `precip`. Defaults to None.
        min_area (float, optional): Minimum contributing area (m2) for streams.
        Defaults to 1e6.
        min_length (float, optional): Minimum stream length. Defaults to 0.
        mexp (Union[float, None], optional): Montgomery exponent for r.stream.extract.
        Defaults to None.
        snap_distance (float, optional): Distance (m) to search for a receiving
        cell for each inflow. Defaults to 250.
        inflow_dst (Union[str, None], optional): Output GeoPackage of snapped inflow
        points for review. Defaults to None.
        memory (Union[int, None], optional): Memory (MB) for r.stream.extract.
        Defaults to 4096 MB.
    """
    if (precip is None) != (mean_precip_dst is None):
        raise ValueError("precip and mean_precip_dst must be specified together")

    r = Raster(dem)
    cell_area = r.csx * r.csy
    min_area_cells = area_to_cells(dem, min_area)

    with TemporaryDirectory(dir=TMP_DIR) as wkdir:
        p = lambda name: os.path.join(wkdir, f"{name}.tif")

        # Flow directions without inflows, used to snap inflows and seed accumulation
        _r_stream_extract(
            dem,
            p("streams_local"),
            p("direction_local"),
            min_area_cells,
            min_length,
            mexp=mexp,
            memory=memory,
            as_cog=False,
        )

        if precip is not None and not Raster(precip).matches(p("direction_local")):
            warp_like(
                precip,
                p("precip"),
                p("direction_local"),
                resample_method="bilinear",
                as_cog=False,
            )
            precip = p("precip")

        seeds = _snap_inflows(
            dem, p("direction_local"), inflows, min_area, snap_distance
        )

        _seeded_weights(
            dem,
            p("direction_local"),
            seeds,
            p("area_weight"),
            precip=precip,
            precip_dst=p("precip_weight"),
        )

        # Corrected accumulation (in cells) for stream extraction
        FlowAccumulation(p("direction_local")).calculate(
            p("area_weight"), p("area_local")
        )
        to_raster(
            from_raster(p("area_local")) / cell_area,
            p("direction_local"),
            p("accumulation"),
            as_cog=False,
        )

        _r_stream_extract(
            dem,
            stream_dst,
            direction_dst,
            min_area_cells,
            min_length,
            accumulation=p("accumulation"),
            mexp=mexp,
            memory=memory,
        )

        # Accumulate on the final flow direction so outputs are consistent with streams
        fa = FlowAccumulation(direction_dst)
        fa.calculate(p("area_weight"), p("area_sum"))

        area_sum = from_raster(p("area_sum"))
        area_sum = da.ma.masked_where(area_sum <= 0, area_sum)

        to_raster(
            (area_sum / 1e6).astype(np.float32), direction_dst, contributing_area_dst
        )

        if precip is not None:
            fa.calculate(p("precip_weight"), p("precip_sum"))
            to_raster(
                (from_raster(p("precip_sum")) / area_sum).astype(np.float32),
                direction_dst,
                mean_precip_dst,
            )

    if inflow_dst is not None:
        properties = {"area_km2": "float"}
        if precip is not None:
            properties["precip"] = "float"

        with fiona.open(
            inflow_dst,
            "w",
            driver="GPKG",
            crs=r.wkt,
            schema={"geometry": "Point", "properties": properties},
        ) as dst:
            dst.writerecords(
                {
                    "geometry": mapping(Point(seed[2], seed[3])),
                    "properties": {
                        "area_km2": seed[0] / 1e6,
                        **(
                            {"precip": seed[1] / seed[0] if seed[0] > 0 else None}
                            if precip is not None
                            else {}
                        ),
                    },
                }
                for seed in seeds.values()
            )


def extract_streams_expanded(
    dem: str,
    stream_dst: str,
    direction_dst: str,
    contributing_area_dst: str,
    precip: Union[str, None] = None,
    mean_precip_dst: Union[str, None] = None,
    min_area: float = 1e6,
    min_length: float = 0,
    mexp: Union[float, None] = None,
    regional_dem: Union[str, list] = MRDEM_DTM,
    regional_cell_size: Union[float, None] = None,
    max_regional_cells: int = 25_000_000,
    buffer: float = 2000.0,
    snap_distance: float = 250.0,
    inflow_dst: Union[str, None] = None,
    memory: Union[int, None] = 4096,
):
    """Extract streams and contributing area (and optionally mean upstream
    precipitation) from a DEM that does not extend to the edges of its watershed(s).
    This extends `extract_streams` by accounting for area that drains into the DEM.

    The area draining into the DEM is calculated using a regional DEM (Canada's
    MRDEM by default) and is used to seed flow accumulation on the DEM:

    1. HydroBASINS level 12 basins that intersect valid data in the DEM, and all
    basins upstream of them, define the search area (see
    `hydrotools.data.upstream_basins`).
    2. The search area is screened at a coarse resolution to find cells that drain
    into the DEM.
    3. Flow is accumulated on the screened area at `regional_cell_size`, and the area
    (and precipitation) flowing into the DEM is collected where flow crosses into it.
    4. Inflows are snapped to the DEM channels and used to seed a weighted flow
    accumulation, which is used to extract streams (see
    `extract_streams_with_inflows`).

    Note, MRDEM extends only a short distance south of the Canada/US border. Include
    additional DEM sources through `regional_dem` for watersheds that drain from
    further south.

    Args:
        dem (str): Digital Elevation Model raster in a projected (preferably
        equal-area) spatial reference with units of metres.
        stream_dst (str): Output streams raster.
        direction_dst (str): Output flow direction raster.
        contributing_area_dst (str): Output contributing area raster (km2).
        precip (Union[str, None], optional): Mean annual precipitation raster that
        covers the DEM and the area draining into it. Gaps are filled with the nearest
        value. Defaults to None.
        mean_precip_dst (Union[str, None], optional): Output mean upstream
        precipitation raster. Required with `precip`. Defaults to None.
        min_area (float, optional): Minimum contributing area (m2) for streams.
        Defaults to 1e6.
        min_length (float, optional): Minimum stream length. Defaults to 0.
        mexp (Union[float, None], optional): Montgomery exponent for r.stream.extract.
        Defaults to None.
        regional_dem (Union[str, list], optional): Regional DEM source(s) readable by
        GDAL. Where sources overlap, later sources take priority. Defaults to MRDEM.
        regional_cell_size (Union[float, None], optional): Cell size (m) used to
        accumulate regional flow. Defaults to the regional DEM resolution, or larger
        if needed to stay within `max_regional_cells`.
        max_regional_cells (int, optional): Maximum number of cells in the regional
        grid when `regional_cell_size` is not specified. Defaults to 25,000,000.
        buffer (float, optional): Distance (m) used to expand the search areas to
        account for differences between datasets. Defaults to 2000.
        snap_distance (float, optional): Distance (m) to search for a receiving
        cell for each inflow. Defaults to 250.
        inflow_dst (Union[str, None], optional): Output GeoPackage of snapped inflow
        points for review. Defaults to None.
        memory (Union[int, None], optional): Memory (MB) for r.stream.extract.
        Defaults to 4096 MB.
    """
    if (precip is None) != (mean_precip_dst is None):
        raise ValueError("precip and mean_precip_dst must be specified together")

    if isinstance(regional_dem, str):
        regional_dem = [regional_dem]

    r = Raster(dem)
    dem_crs = CRS.from_wkt(r.wkt)
    if dem_crs.is_geographic:
        raise ValueError("The DEM must use a projected spatial reference")

    footprint = valid_footprint(dem)

    # An equal-area grid centred on the DEM is used for all regional analysis
    lon, lat = footprint.centroid.x, footprint.centroid.y
    laea = (
        f"+proj=laea +lat_0={lat:.6f} +lon_0={lon:.6f} +x_0=0 +y_0=0 +ellps=GRS80 "
        "+units=m +no_defs"
    )

    print("Collecting HydroBASINS upstream of the DEM")
    geoms = [
        transform_geom(g, "EPSG:4326", laea)
        for g in upstream_basins(footprint, dissolve=False) + [footprint]
    ]

    native_cs = native_cell_size(
        f"/vsicurl/{regional_dem[0]}"
        if regional_dem[0].startswith(("http://", "https://"))
        else regional_dem[0]
    )

    with TemporaryDirectory(dir=TMP_DIR) as wkdir:
        # Screen the area draining into the DEM at a coarse resolution
        left, bottom, right, top = shapely.total_bounds(geoms)
        cs = max(native_cs, math.sqrt((right - left) * (top - bottom) / SCREENING_CELLS))
        bounds = _grid_bounds(
            (left - buffer, bottom - buffer, right + buffer, top + buffer), cs
        )
        shape_ = (
            int(round((bounds[3] - bounds[1]) / cs)),
            int(round((bounds[2] - bounds[0]) / cs)),
        )
        basin_mask = features.rasterize(
            ((g, 1) for g in geoms),
            out_shape=shape_,
            transform=from_origin(bounds[0], bounds[3], cs, cs),
            all_touched=True,
            dtype=np.uint8,
        ).astype(bool)
        buffer_cells = int(math.ceil(buffer / cs)) + 1

        print(f"Screening the area draining into the DEM at {cs:.0f} m")
        screen = _regional_flow(
            regional_dem,
            laea,
            bounds,
            cs,
            binary_dilation(basin_mask, iterations=buffer_cells),
            dem,
            wkdir,
            "screen",
        )

        if not screen["target"].any():
            raise ValueError("The DEM could not be located on the regional DEM")

        drains = _drains_to(
            screen["fd"], screen["valid"], screen["target"], FD_OFF_I, FD_OFF_J
        )

        no_data = binary_dilation(drains) & basin_mask & ~screen["dem_valid"]
        if no_data.any():
            warning(
                f"The regional DEM has no data over {no_data.sum() * cs**2 / 1e6:.1f} "
                "km2 bordering the area that drains into the DEM. Contributing areas "
                "will be underestimated if this is land (ex. outside of MRDEM "
                "coverage). Use `regional_dem` to include additional sources."
            )

        screen_mask = binary_dilation(drains, iterations=buffer_cells)

        # Accumulate area (and precipitation) at the regional resolution
        rows, cols = np.nonzero(screen_mask)
        left = bounds[0] + cols.min() * cs
        right = bounds[0] + (cols.max() + 1) * cs
        top = bounds[3] - rows.min() * cs
        bottom = bounds[3] - (rows.max() + 1) * cs

        if regional_cell_size is None:
            regional_cell_size = max(
                native_cs,
                math.sqrt((right - left) * (top - bottom) / max_regional_cells),
            )
        cs = regional_cell_size
        bounds = _grid_bounds((left, bottom, right, top), cs)
        shape_ = (
            int(round((bounds[3] - bounds[1]) / cs)),
            int(round((bounds[2] - bounds[0]) / cs)),
        )
        transform = from_origin(bounds[0], bounds[3], cs, cs)

        mask = np.zeros(shape_, np.uint8)
        reproject(
            screen_mask.astype(np.uint8),
            mask,
            src_transform=screen["transform"],
            src_crs=laea,
            dst_transform=transform,
            dst_crs=laea,
            resampling=Resampling.nearest,
        )
        del screen, drains, screen_mask

        print(f"Accumulating regional flow at {cs:.0f} m over {shape_} cells")
        regional = _regional_flow(
            regional_dem, laea, bounds, cs, mask.astype(bool), dem, wkdir, "regional"
        )
        valid = regional["valid"]

        cell_area = cs**2
        weights = np.zeros((1 if precip is None else 2,) + shape_, np.float64)
        weights[0][valid] = cell_area

        regional_precip = None
        if precip is not None:
            regional_precip = os.path.join(wkdir, "regional_precip.tif")
            warp(
                precip,
                regional_precip,
                bounds,
                cs,
                cs,
                laea,
                "Float32",
                "bilinear",
                additional_args=REMOTE_WARP_ARGS,
                as_cog=False,
            )
            with rasterio.open(regional_precip) as ds:
                pr = ds.read(1).astype(np.float64)
                profile = ds.profile.copy()
                missing = ~(ds.read_masks(1) > 0) | ~np.isfinite(pr)

            if missing.all():
                raise ValueError("The precipitation raster does not cover the DEM")

            if (missing & valid).any():
                warning(
                    f"The precipitation raster does not cover "
                    f"{(missing & valid).sum() * cs**2 / 1e6:.1f} km2 of the area "
                    "draining into the DEM, which was filled using the nearest values"
                )
                idx = distance_transform_edt(
                    missing, return_distances=False, return_indices=True
                )
                pr = pr[tuple(idx)]
                del idx

                profile.update(dtype="float32")
                with rasterio.open(regional_precip, "w", **profile) as ds:
                    ds.write(pr.astype(np.float32), 1)

            weights[1][valid] = pr[valid] * cell_area
            del pr

        acc = _accumulate(regional["fd"], valid, weights, FD_OFF_I, FD_OFF_J)
        del weights

        # Collect flow that crosses into the DEM, located at the receiving cell
        target = regional["target"]
        k = np.abs(regional["fd"]).astype(np.int64)
        k[(k < 1) | (k > 8)] = 0

        rows, cols = np.nonzero(valid & ~target & (k > 0))
        to_rows = rows + FD_OFF_I[k[rows, cols]]
        to_cols = cols + FD_OFF_J[k[rows, cols]]
        inside = (
            (to_rows >= 0)
            & (to_rows < shape_[0])
            & (to_cols >= 0)
            & (to_cols < shape_[1])
        )
        rows, cols, to_rows, to_cols = (
            rows[inside],
            cols[inside],
            to_rows[inside],
            to_cols[inside],
        )
        crossing = target[to_rows, to_cols]
        rows, cols, to_rows, to_cols = (
            rows[crossing],
            cols[crossing],
            to_rows[crossing],
            to_cols[crossing],
        )

        receiving, inverse = np.unique(
            to_rows * shape_[1] + to_cols, return_inverse=True
        )
        area = np.bincount(inverse, acc[0, rows, cols], minlength=receiving.size)
        precip_area = (
            np.bincount(inverse, acc[1, rows, cols], minlength=receiving.size)
            if precip is not None
            else np.zeros_like(area)
        )
        del acc

        x = bounds[0] + (receiving % shape_[1] + 0.5) * cs
        y = bounds[3] - (receiving // shape_[1] + 0.5) * cs
        points = transform_points(np.column_stack([x, y]), laea, dem_crs)

        inflows = [
            {"x": pt[0], "y": pt[1], "area": a, "precip": pa / a}
            for pt, a, pa in zip(points, area, precip_area)
            if a > 0
        ]

        print(
            f"Found {len(inflows)} inflows with a total area of "
            f"{area.sum() / 1e6:.1f} km2 (largest {area.max(initial=0) / 1e6:.1f} km2)"
        )

        extract_streams_with_inflows(
            dem,
            inflows,
            stream_dst,
            direction_dst,
            contributing_area_dst,
            precip=regional_precip,
            mean_precip_dst=mean_precip_dst,
            min_area=min_area,
            min_length=min_length,
            mexp=mexp,
            snap_distance=snap_distance,
            inflow_dst=inflow_dst,
            memory=memory,
        )
