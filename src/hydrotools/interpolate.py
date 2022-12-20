from multiprocessing.dummy import Pool as DummyPool
from multiprocessing import cpu_count
from typing import Union

import numpy as np
import dask.array as da
from dask_image.ndmorph import binary_erosion
from numba import njit
from numba.typed import List
from sklearn.neighbors import KNeighborsRegressor, BallTree

from hydrotools.raster import Raster, from_raster, to_raster
from hydrotools.utils import GrassRunner


def raster_filter(
    raster_source: str,
    kernel: Union[np.ndarray, tuple],
    method: str,
    filter_dst: str,
    **kwargs,
):
    """Filter a raster using a defined kernel and method

    Args:
        raster_source (str): Input raster
        kernel (Union[np.ndarray, tuple]): Kernel to create the moving window sample.
        The kernal may be a 2-length tuple to define the size of a rectangular window
        or a boolean numpy array that defines the locations of the sample for the
        window.
        method (str): Method used to apply the filter. Choose one of:
            [
                "sum",
                "min",
                "max",
                "diff",
                "mean",
                "median",
                "std",
                "variance",
                "percentile",
                "quantile"
            ]
            When choosing percentil or quantile, the `q` kwarg should be specified.
        filter_dst (str): Path to the output filtered raster

    Kwargs:
        q: Percentile or quantile to use if `method` is percentile or quantile
    """

    @njit(nogil=True)
    def apply_filter(a, kernel, nodata, i_start, j_start, i_end, j_end, method, q):
        if method == "sum":
            stats_func = np.nansum
        elif method == "min":
            stats_func = np.nanmin
        elif method == "max":
            stats_func = np.nanmax
        elif method == "diff":
            stats_func = lambda x: np.nanmax(x) - np.nanmin(x)
        elif method == "mean":
            stats_func = np.nanmean
        elif method == "median":
            stats_func = np.nanmedian
        elif method == "std":
            stats_func = np.nanstd
        elif method == "variance":
            stats_func = np.nanvar
        elif method == "percentile":
            stats_func = lambda x: np.nanpercentile(x, q)
        elif method == "quantile":
            stats_func = lambda x: np.nanquantile(x, q)
            
        output = np.full(a.shape, nodata, np.float32)

        for i in range(i_start, a.shape[1] - i_end):
            for j in range(j_start, a.shape[2] - j_end):

                sample = np.full(kernel.shape, np.nan, np.float32)
                for k_i in range(kernel.shape[0]):
                    for k_j in range(kernel.shape[1]):
                        if kernel[k_i, k_j]:
                            val = a[0, i + k_i - i_start, j + k_j - j_start]
                            if val != nodata:
                                sample[k_i, k_j] = val

                output_val = stats_func(sample)

                output[0, i, j] = nodata if np.isnan(output_val) else output_val

        return output

    nodata = Raster(raster_source).nodata
    src = from_raster(raster_source)

    if isinstance(kernel, tuple):
        if len(kernel) != 2:
            raise IndexError(
                f"Input kernel {kernel} does not have values for two dimensions"
            )

        kernel = np.ones(kernel, bool)
    else:
        if kernel.ndim != 2:
            raise IndexError(
                f"Input kernel of shape {kernel.shape} does not have two dimensions"
            )

        kernel = np.array(kernel, dtype=bool)

    i_start = int(np.ceil((kernel.shape[0] - 1) / 2.0))
    i_end = kernel.shape[0] - 1 - i_start
    j_start = int(np.ceil(kernel.shape[1] - 1) / 2.0)
    j_end = kernel.shape[1] - 1 - j_start

    depth = {
        0: 0,
        1: i_start,
        2: j_start,
    }

    filter_result = da.map_overlap(
        apply_filter,
        src,
        depth=depth,
        boundary=nodata,
        dtype=np.float32,
        kernel=kernel,
        nodata=nodata,
        i_start=i_start,
        j_start=j_start,
        i_end=i_end,
        j_end=j_end,
        method=method,
        q=kwargs.get("q", 25),
    )

    to_raster(
        da.ma.masked_where(filter_result == nodata, filter_result),
        raster_source,
        filter_dst,
    )


def distance_transform(raster_source: str, distance_dst: str):
    """Calculate a distance transform where cells with values of nodata are given
    values of distance to the nearest cell with valid data.

    Args:
        raster_source (str): Raster used to provide locations of valid data.
        distance_dst (str): Output distance transform raster.
    """
    with GrassRunner(raster_source) as gr:
        gr.run_command(
            "r.grow.distance",
            (raster_source, "input", "raster"),
            input="input",
            distance="output",
        )
        gr.save_raster("output", distance_dst)


@njit
def front_contains(
    x1: float, y1: float, x2: float, y2: float, tx: float, ty: float
) -> bool:
    """Determine if point (tx, ty) is within the front normal to the vector
    (x1, y1) -> (x2, y2).

    Args:
        x1 (float): Point 1 x-coordinate.
        y1 (float): Point 1 y-coordinate.
        x2 (float): Point 2 x-coordinate.
        y2 (float): Point 2 y-coordinate.
        tx (float): Test Point x-coordinate.
        ty (float): Test Point y-coordinate.

    Returns:
        bool: True if the Test Point is within the front, False if outside.
    """
    if x1 == x2:
        return (y2 - y1) >= 0 and ty <= y2 or (y2 - y1) <= 0 and ty >= y2
    if y1 == y2:
        return (x2 - x1) >= 0 and tx <= x2 or (x2 - x1) <= 0 and tx >= x2

    inner_angle = np.arctan((x2 - x1) / (y2 - y1))
    slope_angle = np.sign(inner_angle) * np.pi * 90 / 180 - inner_angle
    fty = y2 + np.cos(slope_angle)
    ftx = x2 - np.sin(slope_angle)

    sign = (tx - x2) * (fty - y2) - (ty - y2) * (ftx - x2)

    return x2 < x1 and sign >= 0 or x2 > x1 and sign <= 0


@njit
def width_transform_task(
    regions: np.ndarray, edges: np.ndarray, csx: float, csy: float
) -> np.ndarray:
    """Compute the approximate widths of regions by searching for bounded edges

    Args:
        regions (np.ndarray): Array of regions (float) bounded by a value of -999.
        edges (np.ndarray): Array of edges (bool) of regions.
        csx (float): Cell size in the x-direction.
        csy (float): Cell size in they y-direction.

    Returns:
        np.ndarray: Array of region widths bounded by -999.
    """
    i_bound, j_bound = regions.shape

    width = regions.copy()
    modals = regions.copy()

    stack = [
        [stack_i, stack_j]
        for stack_i in range(i_bound)
        for stack_j in range(j_bound)
        if edges[stack_i, stack_j]
    ]

    offsets = [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]

    while len(stack) > 0:
        next_i, next_j = stack.pop(0)

        search_stack = List()
        searched = {np.int64(next_i): {np.int64(next_j): True}}
        edge_stack = List([[next_i, next_j]])
        region_stack = List()
        i, j = next_i, next_j

        current_distance = (csx + csy) / 2.0

        while True:
            distances = List()
            terminate = False
            for i_off, j_off in offsets:
                i_nbr = np.int64(i + i_off)
                j_nbr = np.int64(j + j_off)

                if (
                    i_nbr == i_bound
                    or i_nbr < 0
                    or j_nbr == j_bound
                    or j_nbr < 0
                    or regions[i_nbr, j_nbr] == -999
                ):
                    continue

                if i_nbr in searched and j_nbr in searched[i_nbr]:
                    continue
                else:
                    dist = np.sqrt(
                        (np.float64(i_nbr - next_i) * csy) ** 2
                        + (np.float64(j_nbr - next_j) * csx) ** 2
                    )

                    try:
                        searched[i_nbr][j_nbr] = True
                    except:
                        searched[i_nbr] = {j_nbr: True}

                    if edges[i_nbr, j_nbr]:
                        distances.append(dist)
                        if len(region_stack) > 0:
                            # Check if the edge opposes the existing center of mass
                            r_com_i, r_com_j = 0.0, 0.0
                            for _i, _j in region_stack:
                                r_com_i += np.float64(_i)
                                r_com_j += np.float64(_j)
                            denom = np.float64(len(region_stack))
                            r_com_i /= denom
                            r_com_j /= denom

                            e_com_i, e_com_j = 0.0, 0.0
                            for _i, _j in edge_stack:
                                e_com_i += np.float64(_i)
                                e_com_j += np.float64(_j)
                            denom = np.float64(len(edge_stack))
                            e_com_i /= denom
                            e_com_j /= denom

                            if not front_contains(
                                e_com_j, e_com_i, r_com_j, r_com_i, j_nbr, i_nbr
                            ):
                                terminate = True
                        edge_stack.append([i_nbr, j_nbr])
                    else:
                        region_stack.append([i_nbr, j_nbr])
                        if len(search_stack) == 0 or dist >= search_stack[-1][0]:
                            search_stack.append([dist, i_nbr, j_nbr])
                        elif dist <= search_stack[0][0]:
                            search_stack.insert(0, [dist, i_nbr, j_nbr])
                        else:
                            idx = -1
                            while dist <= search_stack[idx][0]:
                                idx -= 1
                            search_stack.insert(idx + 1, [dist, i_nbr, j_nbr])

            if len(distances) > 0:
                current_distance = sum(distances) / len(distances)

            if terminate:
                break

            try:
                _, i, j = search_stack.pop(0)
            except:
                break

        for assign_i, j_dict in searched.items():
            for assign_j, _ in j_dict.items():
                width[assign_i, assign_j] += current_distance
                modals[assign_i, assign_j] += 1

    return np.where(modals > 0, width / modals, -999)


def width_transform(src: str, dst: str):
    """Approximate the width of regions constrained by no data.

    Args:
        src (str): Source raster with regions. Regions are constrained by the raster
        no data value.
        dst (str): Destination raster with continuous values of approximate width
        within regions.
    """
    raster_specs = Raster.raster_specs(src)
    csx, csy = raster_specs["csx"], raster_specs["csy"]

    regions = ~da.ma.getmaskarray(from_raster(src))
    region_eroded = binary_erosion(regions, np.ones((1, 3, 3), dtype=bool)).astype(bool)
    edges = regions & ~region_eroded

    region_array, edges_array = da.compute(
        da.squeeze(da.where(regions, 0, -999).astype("float32")),
        da.squeeze(edges),
    )

    width = da.from_array(
        width_transform_task(region_array, edges_array, csx, csy)[np.newaxis, :]
    )
    del region_array, edges_array

    to_raster(width, src, dst)


@njit
def expand_interpolate_task(
    data: np.ndarray, regions: np.ndarray, csx: float, csy: float
):
    """Interpolator for `expand_interpolate`. Extends data outwards into regions.
    Adds to data in place.

    Args:
        data (np.ndarray): Data with values for interpolation, bounded by NaNs.
        regions (np.ndarray): Regions to constrain interpolation.
        csx (float): Cell size in the x-direction.
        csy (float): Cell size in the y-direction.
    """
    i_bound, j_bound = data.shape

    stack = List()
    for stack_i in range(i_bound):
        for stack_j in range(j_bound):
            if not np.isnan(data[stack_i, stack_j]):
                stack.append([stack_i, stack_j])

    offsets = [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]

    stack_2 = List()
    while len(stack) > 0:
        while len(stack) > 0:
            i, j = stack.pop()
            for i_off, j_off in offsets:
                i_nbr = i + i_off
                j_nbr = j + j_off
                if (
                    i_nbr == i_bound
                    or i_nbr < 0
                    or j_nbr == j_bound
                    or j_nbr < 0
                    or not np.isnan(data[i_nbr, j_nbr])
                    or not regions[i_nbr, j_nbr]
                ):
                    continue

                stack_2.append([i_nbr, j_nbr])
                data[i_nbr, j_nbr] = np.inf

        values = List()
        while len(stack_2) > 0:
            i, j = stack_2.pop()

            interp_accum = 0
            total_distance = 0
            for i_off, j_off in offsets:
                i_nbr = np.int64(i + i_off)
                j_nbr = np.int64(j + j_off)

                if (
                    i_nbr == i_bound
                    or i_nbr < 0
                    or j_nbr == j_bound
                    or j_nbr < 0
                    or np.isnan(data[i_nbr, j_nbr])
                    or np.isinf(data[i_nbr, j_nbr])
                    or not regions[i_nbr, j_nbr]
                ):
                    continue

                inverse_dist = 1 / np.sqrt(
                    (np.float64(i - i_nbr) * csy) ** 2.0
                    + (np.float64(j - j_nbr) * csx) ** 2.0
                )

                interp_accum += data[i_nbr, j_nbr] * inverse_dist
                total_distance += inverse_dist

            if total_distance > 0:
                values.append([i, j, interp_accum / total_distance])
                stack.append([i, j])

        # Assign interpolated values and continue
        for i, j, value in values:
            data[np.int64(i), np.int64(j)] = value


def expand_interpolate(src: str, dst: str, regions: str = None):
    """Expand regions with valid data progressively outwards into regions with no data.

    Args:
        src (str): Source raster with values to expand.
        dst (str): Destination raster with interpolated values.
        regions (str, optional): Raster to constrain interpolation extent. Regions are
        defined by areas with data (constrained by no data). Defaults to None.
    """
    raster_specs = Raster.raster_specs(src)
    csx, csy = raster_specs["csx"], raster_specs["csy"]

    data, regions = da.compute(
        da.squeeze(da.ma.filled(from_raster(src), np.nan)),
        da.squeeze(~da.ma.getmaskarray(from_raster(regions)))
        if regions is not None
        else da.ones(raster_specs["shape"][1:], dtype=bool),
    )

    expand_interpolate_task(data, regions, csx, csy)
    output_data = da.from_array(data)
    output_data = da.ma.masked_where(
        da.isnan(output_data) | da.isinf(output_data), output_data
    )[np.newaxis, :]

    to_raster(output_data, src, dst)


def fill_nodata(raster_source: str, destination: str, method: str = "rst"):
    """Fill regions of no data in a raster with interpolated values.

    Args:
        raster_source (str): Input raster to be interpolated.
        destination (str): Output raster with interpolated values replacing no data.
        method (str, optional): Select one of:
            ["bilinear", "bicubic", "rst"]
        Defaults to "rst".
    """
    if method.lower() not in ["bilinear", "bicubic", "rst"]:
        raise ValueError(
            f"Method {method} not recognized as a valid interpolation method"
        )

    with GrassRunner(raster_source) as gr:
        gr.run_command(
            "r.fillnulls",
            (raster_source, "input", "raster"),
            input="input",
            output="output",
            method=method.lower(),
        )
        gr.save_raster("output", destination)


def fill_stats(
    raster_source: str,
    destination: str,
    method: str = "wmean",
    **kwargs,
):
    """Interpolate small regions with no data in a raster.

    See https://grass.osgeo.org/grass82/manuals/r.fill.stats.html

    Args:
        raster_source (str): Input raster dataset to interpolate
        destination (str): Destination raster dataset
        method (str, optional): Interpolation method. Choose one of:
        [
             "wmean", "mean", "median", mode"
        ]
        Defaults to "linear".

    Kwargs:
        distance (int): Number of cells to fill surrounding regions with data.
        Defaults to 3.
        use_map_units (bool): Interpret the distance as map units and not cells.
        Defaults to False.
        cells (int): Minimum number of cells to use for interpolation sample.
        Defaults to 3.
        smooth (bool): Smooth the grid while interpolating. Defaults to True.
    """
    flags = "s"
    if not kwargs.get("smooth", True):
        flags += "k"
    if kwargs.get("use_map_units", False):
        flags += "m"

    with GrassRunner(raster_source) as gr:
        gr.run_command(
            "r.fill.stats",
            (raster_source, "input", "raster"),
            input="input",
            output="output",
            mode=method,
            distance=kwargs.get("distance", 3),
            cells=kwargs.get("cells", 3),
            flags=flags,
        )

        gr.save_raster("output", destination)


def cubic_spline(
    point_source: str,
    template_raster: str,
    destination_raster: str,
    column: str = None,
    method: str = "bicubic",
):
    """Perform interpolation of a set of vector points into a raster grid
    using a b-spline.

    Args:
        point_source (str): Vector point dataset
        template_raster (str): Raster to collect spatial parameters from
        destination_raster (str): Output raster dataset
        column (str): Column name to use as z-values. Defaults to None, whereby the
        geometry z-values will be used.
        method (str, optional): Spline method. Choose from "bilinear" and "bicubic".
        Defaults to "bicubic".
    """
    with GrassRunner(template_raster) as gr:
        gr.run_command(
            "v.surf.bspline",
            (point_source, "points", "vector"),
            input="points",
            column=column,
            method=method,
            raster_output="interpolated",
        )
        gr.save_raster("interpolated", destination_raster)


class PointInterpolator:
    """Utility for custom interpolation"""

    def __init__(self, obs: np.ndarray, obs_z: np.ndarray, pred: np.ndarray):
        self.obs = np.asarray(obs, dtype="float64")
        self.obs_z = np.asarray(obs_z, dtype="float64")
        self.pred = np.asarray(pred, dtype="float64")

        if obs.shape[0] != obs_z.shape[0]:
            raise ValueError(
                "Number of observed points and observed values (z) do not match"
            )

        if obs.size == 0:
            raise ValueError("At least one observed point required")

    def chunks(self, n_chunks: int, return_obs: bool, *args):
        """Generator for chunking and iterating data points to predict

        Args:
            n_chunks (int): Number of chunks
            return_obs (bool): Include observed data in returned generator

        Yields:
            Generator[np.ndarray]: Chunk of predicted points (with obs and obs_z if required)
        """
        if n_chunks < 1:
            n_chunks = 1
        chunkRange = list(range(0, self.pred.shape[0] + n_chunks, n_chunks))
        for fr, to in zip(chunkRange[:-1], chunkRange[1:-1] + [self.pred.shape[0]]):
            if return_obs:
                yield (self.obs, self.obs_z, np.atleast_2d(self.pred[fr:to])) + args
            else:
                yield np.atleast_2d(self.pred[fr:to])

    def map_chunks(
        self,
        func,
        *args,
        n_chunks: int = 500,
        return_obs: bool = False,
        run_sync: bool = False,
    ) -> np.ndarray:
        """Map chunks of predicted points to an interpolation function

        Args:
            func (callable): Interpolator
            n_chunks (int, optional): Number of chunks to use. Defaults to 500.
            return_obs (bool, optional): Return obs and obs_z for method. Defaults to False.
            run_sync (bool, optional): Run synchronously. Defaults to False.

        Returns:
            np.ndarray: Predicted values
        """
        if run_sync:
            values = [func(a) for a in self.chunks(n_chunks, return_obs, *args)]
        else:
            p = DummyPool(cpu_count())
            values = list(p.imap(func, self.chunks(n_chunks, return_obs, *args)))

        return np.concatenate(values)

    def full_idw(self) -> np.ndarray:
        """Perform idw interpolation using all observed data for every predicted point"""
        return self.map_chunks(full_idw, return_obs=True)

    def idw(self, n_neighbours=3) -> np.ndarray:
        """IDW interpolation"""
        n_neighbours = min(n_neighbours, self.obs_z.size)

        def idw_callable(distance):
            return 1 / (distance**2)

        knn = KNeighborsRegressor(n_neighbors=n_neighbours, weights=idw_callable).fit(
            self.obs, self.obs_z
        )

        return self.map_chunks(knn.predict)

    def bounding_idw(self, n_neighbours=4) -> np.ndarray:
        """Perform IDW interpolation, dynamically selecting a number of neighbours to
        ensure each predicted point is within the bounding box of the neighbours.

        Returns:
            np.ndarray: Array of predicted values
        """
        return bidw_task(self.obs, self.obs_z, self.pred, n_neighbours)

    def mean(self, n_neighbours=3) -> np.ndarray:
        """Perform interpolation using a mean of a number of surrounding points

        Args:
            n_neighbours (int, optional): Number of closest neighbours to use.
            Defaults to 3.

        Returns:
            [np.ndarray]: Array of predicted values
        """
        n_neighbours = min(n_neighbours, self.obs_z.size)

        knn = KNeighborsRegressor(n_neighbors=n_neighbours).fit(self.obs, self.obs_z)

        return self.map_chunks(knn.predict)


def bidw_task(obs, obs_z, pred, n_neighbours):
    obs = obs.copy()
    obs_z = obs_z.copy()
    pred_z = np.full(pred.shape[0], np.nan, "float32")

    def bounds(pred_pnts, obs_pnts):
        top = obs_pnts[..., 1].max(1)
        bottom = obs_pnts[..., 1].min(1)
        left = obs_pnts[..., 0].min(1)
        right = obs_pnts[..., 0].max(1)
        return (
            (pred_pnts[:, 0] <= right)
            & (pred_pnts[:, 0] >= left)
            & (pred_pnts[:, 1] >= bottom)
            & (pred_pnts[:, 1] <= top)
        )

    def idw(obs_values, distances):
        idw_result = np.full(obs_values.shape[0], np.nan, "float32")

        zero_dist = np.any(distances == 0, 1)
        zero_dist_locs = np.argmin(distances[zero_dist], 1)
        idw_result[zero_dist] = obs_values[zero_dist][
            (np.arange(zero_dist_locs.shape[0]), zero_dist_locs)
        ]

        weights = 1 / distances[~zero_dist] ** 2
        idw_result[~zero_dist] = np.sum(
            obs_values[~zero_dist] * (weights / weights.sum(1, keepdims=True)), axis=1
        )

        return idw_result

    pred_remaining = np.isnan(pred_z)
    completeness_tracking = []
    while True:
        bt = BallTree(obs)

        distances, indices = bt.query(pred[pred_remaining], n_neighbours)

        bounded = bounds(pred[pred_remaining], obs[indices])

        locs = np.where(pred_remaining)
        locs = locs[0][bounded]
        pred_z[locs] = idw(obs_z[indices][bounded], distances[bounded])

        pred_remaining = np.isnan(pred_z)
        remaining_points = pred_remaining.sum()

        obs = np.concatenate([obs, pred[locs]])
        obs_z = np.concatenate([obs_z, pred_z[locs]])

        completeness = round(
            ((pred_z.shape[0] - float(remaining_points)) / pred_z.shape[0]) * 100, 2
        )
        completeness_tracking.append(completeness)
        print(f"{completeness}%")
        if len(completeness_tracking) > 2 and len(set(completeness_tracking[-3:])) == 1:
            break

    if remaining_points > 0:
        knn = KNeighborsRegressor(n_neighbors=n_neighbours).fit(obs, obs_z)
        pred_z[pred_remaining] = knn.predict(pred[pred_remaining])

    return pred_z


@njit(nogil=True)
def full_idw(args):
    """Complete outer product idw

    Args:
        args (tuple): (obs, obs_z, pred)
    """
    obs, obs_z, pred = args

    obs_rows = obs.shape[0]
    weights = np.zeros(obs_rows, np.float64)

    pred_rows = pred.shape[0]
    output = np.zeros(pred_rows, np.float64)

    for pred_row in range(pred_rows):
        pred_x = pred[pred_row, 0]
        pred_y = pred[pred_row, 1]

        weights_sum = 0
        coincident = np.nan
        for obs_row in range(obs_rows):
            _x = obs[obs_row, 0]
            _y = obs[obs_row, 1]

            distance = np.sqrt((_x - pred_x) ** 2 + (_y - pred_y) ** 2)

            if distance == 0:
                coincident = obs_z[obs_row]
                break
            else:
                weight = 1 / distance**2

            weights[obs_row] = weight
            weights_sum += weight

        if not np.isnan(coincident):
            output[pred_row] = coincident
        else:
            output[pred_row] = np.sum(obs_z * weights / weights_sum)

    return output
