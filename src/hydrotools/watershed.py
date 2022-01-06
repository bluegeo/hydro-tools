import pickle
import gzip
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as DummyPool
from typing import Union

import numpy as np
from numba import jit

from hydrotools.utils import GrassRunner, indices_to_coords, transform_points
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


def auto_basin(dem: str, min_area: float, basin_dataset: str):
    """Automatically generate basins throughout a dataset that are larger than a minimum area

    Args:
        dem (str): Digital Elevation Model raster
        min_area (float): Minimum basin area to constrain size
        basin_dataset (str): Output raster data containing labeled basins
    """
    # Min Area needs to be converted to cells
    r = Raster(dem)
    min_area_cells = int(np.ceil(min_area / (r.csx * r.csy)))

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
        """
        self.path = idx_path

        with open(idx_path, "rb") as f:
            # Initialize from the header
            header_len = int.from_bytes(f.read(8), "big")
            self.__dict__.update(pickle.loads(f.read(header_len)))

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
        with open(self.path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "big")
            f.seek(header_len)
            idx = pickle.loads(gzip.decompress(f.read()))

        return idx

    @classmethod
    def create_index(
        cls,
        direction_raster: str,
        accumulation_raster: str,
        index_path: str,
        minimum_area: float = 1e6,
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
            accumulation_raster (str): Flow accumulation grid
            index_path (str): Path to create a new index
            minimum_area (float, optional): Minimum area to simulate stream locations.
            Defaults to 1E6.
        """
        domain_spec = Raster(accumulation_raster)

        # Load the rasters into memory because the complexity may exceed benefits from dask
        fa = from_raster(accumulation_raster).compute()
        fd = from_raster(direction_raster).compute()

        # Create a boolean mask of simulated streams
        threshold_cells = minimum_area / (domain_spec.csx * domain_spec.csy)
        streams = fa >= threshold_cells

        # Initialize an array for the stack
        visited = np.zeros(streams.shape, "bool")

        def next_fa():
            candidates = np.where(streams & ~visited)
            try:
                i = np.argmax(fa[candidates])
                return candidates[0][i], candidates[1][i]
            except ValueError:
                return

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
        coord = next_fa()
        watersheds = []
        while coord is not None:
            i, j = coord
            ci, ni = delineate(fd, streams, i, j, visited)
            watersheds.append((ci, ni))
            coord = next_fa()

        cls.save(index_path, domain_spec, watersheds)

        return cls(index_path)

    def calculate_stats(
        self, raster_source: str, destination: str, output: str = "csv", stats: Union[list, str] = ["min", "max", "sum", "mean"], **kwargs
    ):
        """Use a generated index to calculate stats at stream locations

        Args:
            raster_source (str): Raster dataset to collect stats from
            destination (str): Destination file for results
            output (str, optional): Output type. Choose from "csv", "tif". Defaults to "csv".

        Kwargs:
            apply_async (bool): Calculate asyncronously. Defaults to False.
            output_crs (Union[int, str]): Output coordinate reference system for points.
            Defaults to None, whereby the domain CRS will be used.
        """
        src = Raster(raster_source)

        if any([src.shape != self.shape, src.top != self.top, src.left != self.left]):
            raise ValueError("Input data must spatially match index domain")

        idx = self.idx

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

            return [c[0] for c in ci], _min, _max, _sum, _mean

        if kwargs.get("apply_async", False):
            p = DummyPool(cpu_count())
            try:
                res = p.map(summarize, [(ci, ni) for ci, ni in idx])
                p.close()
                p.join()
            except Exception as e:
                p.close()
                p.join()
                raise e
        else:
            res = [summarize((ci, ni)) for ci, ni in idx]

        if output == "csv":
            table = []
            for coords, _min, _max, _sum, _mean in res:
                y, x = indices_to_coords(
                    ([_i for _i, _j in coords], [_j for _i, _j in coords]),
                    self.fa.top,
                    self.fa.left,
                    self.fa.csx,
                    self.fa.csy,
                )
                if kwargs.get("output_crs", None) is not None:
                    pts = transform_points(
                        list(zip(x, y)), self.fa.projection, kwargs.get("output_crs")
                    )
                    x, y = [pt[0] for pt in pts], [pt[1] for pt in pts]
                _min = _min.tolist()
                _min = [val if val != np.finfo("float32").max else "" for val in _min]
                _max = _max.tolist()
                _max = [val if val != np.finfo("float32").max else "" for val in _max]
                _sum = _sum.tolist()
                _sum = [val if val != np.finfo("float32").max else "" for val in _sum]
                _mean = _mean.tolist()
                _mean = [val if val != np.finfo("float32").max else "" for val in _mean]

                table += list(zip(x, y, _min, _max, _sum, _mean))

            if not destination.lower().endswith(".csv"):
                destination += ".csv"
            with open(destination, "wb") as f:
                f.write(
                    "\n".join([",".join(map(str, line)) for line in table]).encode(
                        "utf-8"
                    )
                )

        elif output == "raster":
            if destination.lower().endswith(".tif"):
                destination = destination[:-4]

            i, j, _min, _max, _sum, _mean = [], [], [], [], [], []

            for coords, min_subset, max_subset, sum_subset, mean_subset in res:
                i += [_i for _i, _j in coords]
                j += [_j for _i, _j in coords]

                _min += min_subset.tolist()
                _max += max_subset.tolist()
                _sum += sum_subset.tolist()
                _mean += mean_subset.tolist()

            coords = (np.array(i, dtype="uint64"), np.array(j, dtype="uint64"))

            for stat, data in zip(
                ["min", "max", "sum", "mean"], [_min, _max, _sum, _mean]
            ):
                a = np.full(self.shape, float_boundary, "float32")
                a[coords] = data

                dst = destination + f"-{stat}.tif"

                r = Raster.empty(
                    dst,
                    self.top,
                    self.left,
                    self.csx,
                    self.csy,
                    self.wkt,
                    self.shape,
                    "float32",
                    float_boundary,
                )
                r[:] = a
