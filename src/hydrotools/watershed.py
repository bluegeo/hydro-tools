from collections import OrderedDict
from copy import deepcopy
import os
import pickle
import gzip
from tempfile import TemporaryDirectory
from typing import Union

import dask.array as da
import numpy as np
from numba import jit
import fiona
from fiona.crs import from_string

from hydrotools.utils import (
    GrassRunner,
    indices_to_coords,
    transform_points,
    proj4_string,
)
from hydrotools.raster import Raster, from_raster


def flow_direction_accumulation(
    dem: str,
    direction_grid: str,
    accumulation_grid: str,
    single: bool = True,
    positive_only: bool = True,
    mem_manage: bool = False,
):
    """Generate Flow Direction and Flow Accumulation grids using GRASS r.watershed

    Args:
        dem (str): Digital Elevation Model raster
        direction_grid (str): Output flow direction raster
        accumulation_grid (str): Output flow accumulation raster
        single (bool, optional): Output Single Flow Direction. Defaults to True.
        positive_only (bool, optional): Only include positive flow direction values.
        Defaults to True.
        mem_manage (bool, optional): Manage memory during execution. Defaults to False.
    """
    flags = ""
    if positive_only:
        flags += "a"
    if single:
        flags += "s"
    if mem_manage:
        flags += "m"

    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.watershed",
            (dem, "dem", "raster"),
            elevation="dem",
            drainage="fd",
            accumulation="fa",
            flags=flags,
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


def auto_basin(dem: str, min_area: float, basin_dataset: str):
    """Automatically generate basins throughout a dataset that are larger than a minimum area

    Args:
        dem (str): Digital Elevation Model raster
        min_area (float): Minimum basin area to constrain size
        basin_dataset (str): Output raster data containing labeled basins
    """
    # Min Area needs to be converted to cells
    min_area_cells = area_to_cells(dem, min_area)

    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.watershed",
            (dem, "dem", "raster"),
            elevation="dem",
            threshold=min_area_cells,
            basin="b",
            flags="s",
        )
        gr.save_raster("b", basin_dataset)


def extract_streams(
    dem: str,
    accumulation_src: str,
    stream_dst: str,
    min_area: float = 1e6,
    min_length: float = 0,
):
    """Extract simulated streams

    Args:
        dem (str): Path to a Digital Elevation Model (DEM)
        accumulation_src (str): Flow Accumulation dataset derived from `flow_direction_accumulation`
        stream_dst (str): Output Streams raster path
        min_area (float, optional): Minimum watershed area for streams. Defaults to 1e6.
        min_length (float, optional): Minimum stream length. Defaults to 0.
    """
    # Min Area needs to be converted to cells
    min_area_cells = area_to_cells(dem, min_area)

    with GrassRunner(dem) as gr:
        gr.run_command(
            "r.stream.extract",
            (dem, "dem", "raster"),
            (accumulation_src, "accum", "raster"),
            elevation="dem",
            accumulation="accum",
            stream_length=float(min_length),
            threshold=min_area_cells,
            stream_raster="streams",
        )
        gr.save_raster("streams", stream_dst)


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

    def __init__(self, idx_path: str, idx=None):
        """Load an existing index from a path

        Args:
            idx_path (str): Path to previously created index
            idx (list): Existing index, which avoids the need to re-read the index from
            disk. This argument is reserved for the `create_index` classmethod and 
            should not be used directly without caution.
        """
        self.path = idx_path

        with open(idx_path, "rb") as f:
            # Initialize from the header
            header_len = int.from_bytes(f.read(8), "big")
            self.__dict__.update(pickle.loads(f.read(header_len)))

        self.inmem_idx = idx

    @staticmethod
    def save(path, domain, idx):
        """Save the index and parameters to a file

        Args:
            path (str): Path of the output file
            domain (dict): Specifications of the index domain
            (read from the accumulation dataset)
            idx (list): Index tree (nested lists)
        """
        with open(path, "wb") as f:
            header = pickle.dumps(domain)
            header_len = len(header)
            seek_bytes = header_len.to_bytes(8, "big")

            f.write(seek_bytes)
            f.write(header)
            f.write(gzip.compress(pickle.dumps(idx)))

    @property
    def idx(self) -> list:
        """Load the index from the initialization file

        Returns:
            list: Watershed index graph of nested lists
        """
        if self.inmem_idx is not None:
            return self.inmem_idx

        with open(self.path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "big")
            f.seek(header_len + 8)
            idx = pickle.loads(gzip.decompress(f.read()))

            self.inmem_idx = idx

        return idx

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

        @jit(nopython=True)
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

        @jit(nopython=True)
        def delineate(fd, streams, i, j, visited):
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
                # Collect a new element to test
                if ci_e[cursor] < len(ci[cursor]):
                    i, j = ci[cursor][ci_e[cursor]]
                    ci_e[cursor] += 1
                else:
                    # Backtrack or break out of the algo
                    cursor -= 1
                    if cursor < 0:
                        break
                    continue

                # Test the current element at location (i, j)
                stream_elems = []
                for row_offset in range(-1, 2):
                    for col_offset in range(-1, 2):
                        t_i, t_j = i + row_offset, j + col_offset

                        if visited[t_i, t_j]:
                            continue

                        # Check if the element at this offset contributes to the element being tested
                        if fd[t_i, t_j] == directions[row_offset + 1][col_offset + 1]:
                            # This element has now been visited
                            visited[t_i, t_j] = True

                            if streams[t_i, t_j]:
                                # This element comprises a stream - add as a nested element
                                stream_elems.append((t_i, t_j))
                            else:
                                # Add to contributing stack, and the testing queue
                                ci[cursor].append((t_i, t_j))

                # Add nested locations and propagate past any stream elements
                this_index = cursor
                for se in stream_elems:
                    # Add nested to current
                    cursor = len(ci_e)
                    ni[this_index].append(cursor)
                    # New list item
                    ci.append([se])
                    ci_e.append(0)
                    ni.append([-1])

            return ci, [[j for j in i if j != -1] for i in ni]

        # Run the alg
        coord = next_ws()
        watersheds = []
        while coord is not None:
            i, j = coord
            ci, ni = delineate(fd, streams, i, j, visited)
            watersheds.append((ci, ni))
            coord = next_ws()

        cls.save(index_path, domain_spec, watersheds)

        return cls(index_path, idx=watersheds)

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
        for coords, _ in self.idx:
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

        fields = [] if data is None else [(key, "float") for key in data.keys()]

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
                        []
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
        src = Raster(raster_source)

        if any([src.shape != self.shape, src.top != self.top, src.left != self.left]):
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

        for ci, ni in self.idx:
            _min, _max, _sum, _mean = summarize((ci, ni))
            stat_output["min"] += map_results(_min)
            stat_output["max"] += map_results(_max)
            stat_output["sum"] += map_results(_sum)
            stat_output["mean"] += map_results(_mean)

        self.save_locations(
            destination, data=stat_output, output_crs=kwargs.get("output_crs", None)
        )
