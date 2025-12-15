from collections import OrderedDict
from copy import deepcopy
import os
import pickle
import gzip
from sysconfig import get_path
from tempfile import TemporaryDirectory
from typing import Union

import dask.array as da
import numpy as np
from numba import njit
import fiona
from fiona.crs import from_string
from skimage.measure import label as image_label

from hydrotools.morphology import stream_slope
from hydrotools.utils import (
    GrassRunner,
    indices_to_coords,
    transform_points,
    proj4_string,
)
from hydrotools.raster import (
    Raster,
    from_raster,
    to_raster,
    TempRasterFile,
    TempRasterFiles,
    vectorize,
    warp_like,
    vector_to_raster,
)


def condition_dem(dem: str, dem_cnd_dst: str):
    """Apply a hydrological conditioning to remove sinks.

    Args:
        dem (str): Digital Elevation Model raster.
        dem_cnd_dst (str): Output conditioned Digital Elevation Model.
    """
    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.hydrodem", (dem, "dem", "raster"), input="dem", output="dem_cnd"
        )
        gr.save_raster("dem_cnd", dem_cnd_dst)


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
    only_topology: bool = False,
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
        only_topology (bool): Do not run stream order, only vectorize the streams and
        add topology. Defaults to False. **Note** this requires a modified version of
        the GRASS r.stream.order tool.
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
            "r.stream.topology" if only_topology else "r.stream.order",
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
                    input="direction",
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
                input="direction",
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


def stream_analysis(
    dem: str,
    precipitation: str,
    lakes: str,
    min_area: float = 1e6,
    min_length: float = 25,
    **kwargs,
):
    """
    Run a watershed analysis by extracting streams from a DEM and applying
    attributes to stream reaches.

    Args:
        dem (str): Path to a Digital Elevation Model raster
        min_area (float, optional): Minimum watershed area for streams. Defaults to 1e6.
        min_length (float, optional): Minimum stream length. Defaults to 25.
        lakes (str | None, optional): Path to a lakes vector file to mask out lakes from
        the watershed analysis. Defaults to None.
        precipitation (str | None, optional): Path to a precipitation raster file. Defaults to None.
    """
    with TempRasterFiles(12) as (
        fd_tmp,
        fd,
        fa,
        fa_scaled,
        streams,
        lakes_src,
        precip_src,
        area,
        mean_precip,
        bfw,
        gradient,
        reaches
    ):
        flow_direction_accumulation(
            dem,
            fd_tmp,
            fa,
        )

        fa_data = fa_data = from_raster(fa).astype("float32")

        # Weight flow accumulation by precipitation
        warp_like(precipitation, precip_src, dem, as_cog=False)

        precip = from_raster(precip_src)
        precip_scaled = da.clip(
            precip, 0, kwargs.get("precip_max_mm", 3000)
        ) / kwargs.get("precip_max_mm", 3000)

        fa_data *= precip_scaled

        vector_to_raster(
            lakes, dem, lakes_src, as_mask=True, all_touched=True, as_cog=False
        )

        lakes_a = ~da.ma.getmaskarray(from_raster(lakes_src))

        fa_data = da.where(lakes_a, 0, fa_data)

        fa_data = da.ma.masked_where(da.ma.getmaskarray(from_raster(fa)), fa_data)

        to_raster(fa_data, fa, fa_scaled, as_cog=False)

        extract_streams(
            dem,
            fa_scaled,
            streams,
            fd,
            min_area=min_area,
            min_length=min_length,
        )

        # -------------------------
        # Bankfull width estimation
        # -------------------------
        flow_accum = FlowAccumulation(fd)
        flow_accum.contributing_area(area)
        flow_accum.calculate(precip, mean_precip, "mean")

        precip_a = from_raster(mean_precip) / 10  # mm -> cm

        ca_a = from_raster(area)

        bankfull_width = (
            kwargs.get("bw_coeff", 0.042)
            * (ca_a ** kwargs.get("a_exp", 0.48))
            * (precip_a ** kwargs.get("p_exp", 0.74))
        )

        # Isolate streams
        streams_a = from_raster(streams)

        to_raster(
            da.ma.masked_where(da.ma.getmaskarray(streams_a), bankfull_width),
            streams,
            bfw,
        )

        # -------------------------
        # Stream Gradient
        # -------------------------
        stream_slope(
            dem,
            streams,
            gradient,
            focal_mean_dist=kwargs.get("gradient_focal_mean_dist", 25),
        )

        # -------------------------
        # Reach Break Classification
        # ------------------------
        grad_a = from_raster(gradient)
        gradient_breaks = np.arange(0, 90, kwargs.get("gradient_break_interval", 1))
        grad_classes = da.digitize(grad_a, gradient_breaks) + 1

        bfw_data = from_raster(bfw)
        bfw_breaks = np.arange(0, 100, kwargs.get("bfw_break_interval", 1))
        bfw_classes = da.digitize(bfw_data, bfw_breaks) + 1

        # Combine the classes to create unique reaches
        reach_classes = (
            streams_a.astype("uint64") * 1000000
            + bfw_classes.astype("uint64") * 1000
            + grad_classes.astype("uint64") * 10
            + lakes_a.astype("uint64")
        )

        new_stream_labels = image_label(
            da.where(da.ma.getmaskarray(streams_a), 0, reach_classes).compute(),
            connectivity=2,
        )

        reach_labels = da.from_array(
            new_stream_labels,
            chunks=streams_a.chunks,
        )

        to_raster(
            da.ma.masked_where(
                da.ma.getmaskarray(streams_a),
                reach_labels.astype("uint32"),
            ),
            streams,
            reaches,
            nodata_value=0,
        )

        # -------------------------
        # Reach Break Classification
        # ------------------------


class WatershedIndex:
    """
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
