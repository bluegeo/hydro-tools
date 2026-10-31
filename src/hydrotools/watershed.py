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
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from numba import njit
import fiona
from fiona.crs import from_string

from hydrotools.config import TMP_DIR
from hydrotools.data import HYDROBASINS_REGIONS, Inflows, valid_footprint
from hydrotools.flow import TiledAccumulation, extract_streams as extract_stream_cells
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
)


def flow_direction_accumulation(
    dem: str,
    direction_grid: str,
    accumulation_grid: str,
    single: bool = True,
    positive_only: bool = True,
    memory: Union[int, None] = None,
    beautify: Union[bool, None] = None,
    as_cog: bool = True,
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
        as_cog (bool, optional): Save outputs as COGs. Defaults to True.
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
        gr.save_raster("fd", direction_grid, as_cog=as_cog)
        gr.save_raster("fa", accumulation_grid, as_cog=as_cog)


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


def flow_direction(dem: str, dst: str, memory: Union[int, None] = 4096):
    """Generate a single flow direction grid using GRASS r.watershed

    Args:
        dem (str): Digital Elevation Model raster
        dst (str): Output flow direction raster
        memory (Union[int, None], optional): Manage memory during execution by assigning
        a maximum block size. Defaults to 4096 MB.
    """
    flags = "s" + ("m" if memory is not None else "")
    kwargs = {"memory": memory} if memory is not None else {}

    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.watershed",
            (dem, "dem", "raster"),
            elevation="dem",
            drainage="fd",
            flags=flags,
            **kwargs,
        )
        gr.save_raster("fd", dst, as_cog=False)


class FlowAccumulation:
    """
    Summarize sums and means of raster parameters over the area upstream of every
    cell, using the tiled accumulation in `hydrotools.flow` (memory is bounded by
    `tile_size`, `workers`, and `max_memory`).

    Rasters used in `calculate` are warped to the direction grid as needed.
    """

    def __init__(
        self,
        direction: Union[str, None],
        tile_size: int = 2048,
        workers: Union[int, None] = None,
        max_memory: int = 4096,
    ):
        """Initialize a FlowAccumulation analysis

        Args:
            direction (str): Path to a flow direction raster generated using
            r.watershed or r.stream.extract (1-8 counterclockwise from north-east).
            tile_size (int, optional): Accumulation tile size in cells. Defaults to
            2048.
            workers (Union[int, None], optional): Tiles accumulated concurrently.
            Defaults to the number of CPUs, limited by `max_memory`.
            max_memory (int, optional): Approximate limit (MB) on the memory used by
            accumulation tiles. Defaults to 4096.
        """
        self.wkdir = TemporaryDirectory(dir=TMP_DIR)
        self.engine_kwargs = {
            "tile_size": tile_size,
            "workers": workers,
            "max_memory": max_memory,
        }
        self._modals = None
        self._engine = None
        self._tmp_count = 0
        self.direction = direction

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction: Union[str, None]):
        self._direction = direction
        self._engine = None
        self._modals = None

    @property
    def engine(self) -> TiledAccumulation:
        if self.direction is None:
            raise ValueError("A flow direction grid is required")
        if self._engine is None:
            self._engine = TiledAccumulation(self.direction, **self.engine_kwargs)
        return self._engine

    def _tmp(self, name: str) -> str:
        """A unique path in the working directory"""
        self._tmp_count += 1

        return os.path.join(self.wkdir.name, f"{name}_{self._tmp_count}.tif")

    def _seeds(self, raster_source: Union[str, None] = None) -> Union[list, None]:
        """Flow added at cells, in cell units, as (row, col, [value]). The value is a
        number of cells when `raster_source` is None, otherwise the sum of the raster
        over those cells. Overridden to add flow from outside of the grid."""
        return None

    @property
    def cell_area(self) -> float:
        r = Raster(self.direction)
        return float(r.csx * r.csy)

    @property
    def modals(self) -> str:
        """Raster with the number of cells upstream of (and including) each cell"""
        if self._modals is None:
            self._modals = self._tmp("modals")
            self.engine.accumulate(
                [1.0], [self._modals], seeds=self._seeds(), dtype="float64"
            )

        return self._modals

    def calculate(self, raster_source: str, dst: str, method: str = "sum"):
        """Calculate flow accumulation statistics

        Args:
            raster_source (str): Path to a raster dataset to calculate stats on. It is
            warped to the direction grid as needed.
            dst (str): Path to save the output raster
            method (str, optional): "sum" or "mean". Defaults to "sum".
        """
        if method not in ("sum", "mean"):
            raise ValueError("method must be one of 'sum' or 'mean'")

        sum_result = dst if method == "sum" else self._tmp("sum_result")
        self.engine.accumulate(
            [raster_source],
            [sum_result],
            seeds=self._seeds(raster_source),
            dtype="float64",
        )

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
        ca = (from_raster(self.modals) * self.cell_area) / 1e6

        to_raster(da.ma.masked_where(ca == 0, ca), self.direction, dst)

    def __del__(self):
        """Clean up temporary working directory"""
        self.wkdir.cleanup()


# Row/col offsets indexed by r.watershed flow direction (1-8 counterclockwise from NE)
FD_OFF_I = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0])
FD_OFF_J = np.array([0, 1, 0, -1, -1, -1, 0, 1, 1])


def _snap(accumulation: str, points: list, snap_distance: float) -> list:
    """Snap points to the cell with the largest accumulation within a distance

    Returns:
        list: (row, col) for each point, or None where there are no valid cells
    """
    cells = []
    with rasterio.open(accumulation) as ds:
        rad_i = int(math.ceil(snap_distance / abs(ds.res[1])))
        rad_j = int(math.ceil(snap_distance / abs(ds.res[0])))
        ii, jj = np.mgrid[-rad_i : rad_i + 1, -rad_j : rad_j + 1]
        outside = np.hypot(ii * ds.res[1], jj * ds.res[0]) > snap_distance

        for x, y in points:
            row, col = ds.index(x, y)
            window = Window(col - rad_j, row - rad_i, 2 * rad_j + 1, 2 * rad_i + 1)
            a = np.ma.masked_where(
                outside, ds.read(1, window=window, masked=True, boundless=True)
            )
            if np.ma.getmaskarray(a).all():
                cells.append(None)
                continue

            i, j = np.unravel_index(np.ma.argmax(a), a.shape)
            cells.append((row - rad_i + int(i), col - rad_j + int(j)))

    return cells


class ExpandedFlowAccumulation(FlowAccumulation):
    """
    Flow accumulation (see `FlowAccumulation`) on a DEM that does not extend to the
    edges of its watershed(s). Flow from outside of the DEM is found using HydroRIVERS
    and HydroBASINS (see `hydrotools.data.Inflows`), snapped to the DEM, and included
    in all results.

    1. Initialize using a DEM

        `efa = ExpandedFlowAccumulation(dem_path)`

    2. Extract streams

        `efa.extract_streams(stream_dst, direction_dst)`

    3. Calculate contributing area and other statistics

        `efa.contributing_area(contributing_area_dst)`

        `efa.calculate(precip_path, mean_precip_dst, method="mean")`

    Rasters used in `calculate` should cover the area draining into the DEM. Where
    they do not, the value at the inflow location is used.
    """

    def __init__(
        self,
        dem: str,
        direction: Union[str, None] = None,
        snap_distance: float = 250.0,
        regions: tuple = HYDROBASINS_REGIONS,
        memory: Union[int, None] = 4096,
        **kwargs,
    ):
        """Calculate the flow into a DEM

        Args:
            dem (str): Digital Elevation Model raster in a projected spatial reference
            with units of metres.
            direction (Union[str, None], optional): Flow direction raster for the DEM
            (see `flow_direction`). Defaults to None, where it is generated using
            r.watershed.
            snap_distance (float, optional): Distance (m) to search for a receiving
            cell on the DEM for each inflow. Defaults to 250.
            regions (tuple, optional): HydroSHEDS regions to search for inflows.
            Defaults to `hydrotools.data.HYDROBASINS_REGIONS`.
            memory (Union[int, None], optional): Manage memory during r.watershed by
            assigning a maximum block size. Defaults to 4096 MB.

        kwargs:
            Arguments for `FlowAccumulation` (`tile_size`, `workers`, `max_memory`).
        """
        super().__init__(None, **kwargs)

        self.dem = dem
        self.snap_distance = snap_distance
        self._cells = None

        if direction is None:
            print("Calculating flow direction")
            direction = os.path.join(self.wkdir.name, "direction.tif")
            flow_direction(dem, direction, memory=memory)
        self.direction = direction

        print("Finding inflows using HydroRIVERS")
        self.inflows = Inflows(valid_footprint(dem), regions)
        print(
            f"Found {len(self.inflows)} inflows with a total area of "
            f"{self.inflows.area.sum() / 1e6:.1f} km2"
        )

    @property
    def cells(self) -> list:
        """The (row, col) receiving each inflow, or None where an inflow could not be
        snapped to the DEM"""
        if self._cells is None:
            points = transform_points(
                self.inflows.points, "EPSG:4326", Raster(self.direction).wkt
            )
            # Snap to the largest accumulation (without inflows) near each inflow
            unseeded = self._tmp("unseeded")
            self.engine.accumulate([1.0], [unseeded], dtype="float64")
            self._cells = _snap(unseeded, points, self.snap_distance)

            lost = self.inflows.area[[c is None for c in self._cells]].sum()
            if lost > 0:
                warning(
                    f"{lost / 1e6:.3f} km2 of inflow could not be snapped to the DEM "
                    "and was not included"
                )

        return self._cells

    def _seeds(self, raster_source: Union[str, None] = None) -> list:
        cells = self.cells
        values = self.inflows.area / self.cell_area

        if raster_source is not None:
            means = self.inflows.mean(raster_source)
            if np.isnan(means).any():
                # Use the value on the DEM where the raster does not cover the inflow
                with rasterio.open(self.direction) as ds:
                    transform, crs, shape = ds.transform, ds.crs, ds.shape
                with rasterio.open(raster_source) as src, WarpedVRT(
                    src, crs=crs, transform=transform, width=shape[1], height=shape[0]
                ) as vrt:
                    for n, cell in enumerate(cells):
                        if np.isnan(means[n]) and cell is not None:
                            row, col = cell
                            means[n] = vrt.read(
                                1, window=Window(col, row, 1, 1), masked=True
                            ).filled(np.nan)[0, 0]
            means = np.nan_to_num(means)
            values = values * means

        return [
            (cell[0], cell[1], [value])
            for cell, value in zip(cells, values)
            if cell is not None
        ]

    def extract_streams(
        self,
        stream_dst: str,
        direction_dst: str,
        min_area: float = 1e6,
        min_length: float = 0,
        mexp: float = None,
    ):
        """Extract streams where the contributing area, including inflows, reaches a
        minimum area. Streams are labelled with unique segment ids.

        Args:
            stream_dst (str): Output streams raster path
            direction_dst (str): Output flow direction raster path
            min_area (float, optional): Minimum contributing area (m2) for streams.
            Defaults to 1e6.
            min_length (float, optional): First-order streams shorter than this
            length (m) are removed. Defaults to 0.
            mexp (float, optional): Not supported. Defaults to None.
        """
        if mexp is not None:
            warning("mexp is not supported by ExpandedFlowAccumulation and is ignored")

        print("Extracting streams")
        extract_stream_cells(
            self.modals,
            self.direction,
            stream_dst,
            min_area / self.cell_area,
            min_length,
            tile_size=self.engine.tile_size,
        )
        translate_to_cog(self.direction, direction_dst)

    def save_inflows(self, dst: str):
        """Save snapped inflows to a GeoPackage for review

        Args:
            dst (str): Output GeoPackage path
        """
        r = Raster(self.direction)
        records = self.inflows.records()

        with fiona.open(
            dst,
            "w",
            driver="GPKG",
            crs=r.wkt,
            schema={
                "geometry": "Point",
                "properties": {
                    "hyriv_id": "int",
                    "area_km2": "float",
                    "subtracted_km2": "float",
                    "upstream_reaches": "int",
                    "upstream_polygons": "int",
                    "snapped": "bool",
                },
            },
        ) as out:
            for cell, rec, (x, y) in zip(
                self.cells,
                records,
                transform_points(self.inflows.points, "EPSG:4326", r.wkt),
            ):
                if cell is not None:
                    x, y = r.left + (cell[1] + 0.5) * r.csx, r.top - (cell[0] + 0.5) * r.csy
                out.write(
                    {
                        "geometry": {"type": "Point", "coordinates": (x, y)},
                        "properties": {
                            "hyriv_id": rec["hyriv_id"],
                            "area_km2": rec["area_km2"],
                            "subtracted_km2": rec["subtracted_km2"],
                            "upstream_reaches": rec["upstream_reaches"],
                            "upstream_polygons": rec["upstream_polygons"],
                            "snapped": cell is not None,
                        },
                    }
                )


def extract_streams_expanded(
    dem: str,
    stream_dst: str,
    direction_dst: str,
    contributing_area_dst: str,
    min_area: float = 1e6,
    min_length: float = 0,
    inflow_dst: Union[str, None] = None,
    **kwargs,
) -> ExpandedFlowAccumulation:
    """Extract streams and contributing area from a DEM that does not extend to the
    edges of its watershed(s). See `ExpandedFlowAccumulation`.

    Args:
        dem (str): Digital Elevation Model raster in a projected spatial reference with
        units of metres.
        stream_dst (str): Output streams raster path.
        direction_dst (str): Output flow direction raster path.
        contributing_area_dst (str): Output contributing area raster path (km2).
        min_area (float, optional): Minimum contributing area (m2) for streams.
        Defaults to 1e6.
        min_length (float, optional): Minimum stream length (m). Defaults to 0.
        inflow_dst (Union[str, None], optional): Output GeoPackage of snapped inflow
        points for review. Defaults to None.

    kwargs:
        Arguments for `ExpandedFlowAccumulation`

    Returns:
        ExpandedFlowAccumulation: Use to calculate other statistics on the streams
    """
    efa = ExpandedFlowAccumulation(dem, **kwargs)
    efa.extract_streams(stream_dst, direction_dst, min_area, min_length)
    efa.contributing_area(contributing_area_dst)

    if inflow_dst is not None:
        efa.save_inflows(inflow_dst)

    return efa



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
