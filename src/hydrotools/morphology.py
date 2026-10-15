import math
import os
from collections import OrderedDict
from logging import warning
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from tempfile import TemporaryDirectory

import dask.array as da
import fiona
import numpy as np
from dask_image.ndmorph import binary_erosion
from numba import njit
from numba.typed import List
from scipy.interpolate import griddata
from scipy.optimize import minimize
from shapely.geometry import LineString, Point
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import BallTree

from hydrotools.config import TMP_DIR
from hydrotools.elevation import slope
from hydrotools.interpolate import (
    distance_transform,
    raster_filter,
)
from hydrotools.raster import (
    Raster,
    from_raster,
    raster_where,
    to_raster,
    vectorize,
    warp_like,
)
from hydrotools.utils import (
    GrassRunner,
    TempRasterFile,
    TempRasterFiles,
    compare_projections,
    kernel_from_distance,
)
from hydrotools.watershed import (
    FlowAccumulation,
    extract_streams,
    flow_direction_accumulation,
)

DEFAULT_COST_FLOOR = math.tan(math.radians(0.5))
"""Minimum friction (m/m) applied to the rise cost surface on flat terrain."""


@njit
def _upstream_compute(
    values_arr,
    streams_mask,
    fd_arr,
    gradient_arr,
    fd_off_i,
    fd_off_j,
    csx,
    csy,
    distance,
    max_window,
):
    # upstream_dirs[di+1][dj+1] = the fd value a neighbour at offset (di, dj) must
    # have to drain into the current cell — used to identify inlet cells.
    upstream_dirs = [[7, 6, 5], [8, 0, 4], [1, 2, 3]]
    diag = (csx**2 + csy**2) ** 0.5
    n_rows = values_arr.shape[0]
    n_cols = values_arr.shape[1]

    # Circular buffers reused across inlet traversals.
    # dist_buf stores the cumulative distance from the inlet to each entry so that
    # the pop condition (current_cum_dist - dist_buf[head] > distance) correctly
    # represents how far the oldest buffered cell is from the current cell.
    val_buf = np.empty(max_window, dtype=np.float64)
    dist_buf = np.empty(max_window, dtype=np.float64)

    for i in range(n_rows):
        for j in range(n_cols):
            if not streams_mask[i, j]:
                continue

            # Identify inlet cells: stream cells with no upstream stream neighbours
            is_inlet = True
            for di in range(-1, 2):
                if not is_inlet:
                    break
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if ni < 0 or ni >= n_rows or nj < 0 or nj >= n_cols:
                        continue
                    if (
                        streams_mask[ni, nj]
                        and fd_arr[ni, nj] == upstream_dirs[di + 1][dj + 1]
                    ):
                        is_inlet = False
                        break

            if not is_inlet:
                continue

            # Seed the window with the inlet cell itself (cumulative distance = 0)
            head = 0
            tail = 1
            size = 1
            current_cum_dist = 0.0
            val_buf[0] = values_arr[i, j]
            dist_buf[0] = 0.0
            val_sum = values_arr[i, j]

            ci, cj = i, j

            while True:
                fd_val = fd_arr[ci, cj]
                if fd_val <= 0 or fd_val > 8:
                    break

                ni = ci + fd_off_i[fd_val]
                nj = cj + fd_off_j[fd_val]

                if ni < 0 or ni >= n_rows or nj < 0 or nj >= n_cols:
                    break
                if not streams_mask[ni, nj]:
                    break

                step_dist = (
                    diag
                    if (fd_off_i[fd_val] != 0 and fd_off_j[fd_val] != 0)
                    else (csx if fd_off_j[fd_val] != 0 else csy)
                )
                current_cum_dist += step_dist

                # Push the downstream cell's value with its cumulative distance
                val_buf[tail] = values_arr[ni, nj]
                dist_buf[tail] = current_cum_dist
                tail = (tail + 1) % max_window
                size += 1
                val_sum += values_arr[ni, nj]

                # Pop entries whose distance from the current cell exceeds the window.
                # The newest entry is always at distance 0, so it is never popped.
                while current_cum_dist - dist_buf[head] > distance:
                    val_sum -= val_buf[head]
                    head = (head + 1) % max_window
                    size -= 1

                gradient_arr[ni, nj] = val_sum / size

                ci, cj = ni, nj

    return gradient_arr


def sinuosity(stream_vector: str, sinuosity_vector: str, sampling: float | str = 100):
    """Calculate sinuosity over a moving window using a line vector file

    Args:
        stream_vector (str): Input stream vector dataset.
        Must jave a LineString or MultiLineString geometry
        sinuosity_vector (str): Output vector dataset
        sampling (Union[float, str], optional): A sinuosity sampling distance.
        Should be one of either: a) A regular distance along streams, or b) "reach"
        indicating that each feature is used in its entirety. Defaults to 100.

    Raises:
        ValueError: If the input vector is not a line
    """
    sampling = sampling.lower() if isinstance(sampling, str) else float(sampling)

    if sampling == "reach":
        raise NotImplementedError("Not developed yet")

    def apply(collection):
        geom = collection["geometry"]

        if geom["type"] == "MultiLineString":
            geoms = geom["coordinates"]
        elif geom["type"] == "LineString":
            geoms = [geom["coordinates"]]
        else:
            raise ValueError("Geometry must be of type MultiLineString or LineString")

        sinuosity = []
        for geom in geoms:
            base_line = LineString(geom)

            # Interpolate points over the sample distance
            points = (
                [Point(geom[0])]
                + [
                    base_line.interpolate(dist)
                    for dist in np.arange(
                        1, int(np.ceil(base_line.length / sampling)), dtype="float32"
                    )
                    * sampling
                ]
                + [Point(geom[-1])]
            )

            # Calculate sinuosity for each point
            sinuosity += [
                (
                    p.x,
                    p.y,
                    p.buffer(sampling / 2.0).intersection(base_line).length / sampling,
                )
                for p in points
            ]

        return sinuosity

    with fiona.open(stream_vector) as vect:
        p = Pool(cpu_count())

        results = p.imap(apply, vect)

        points = []
        for result in results:
            if result:
                points += result

        output_schema = {
            "geometry": "Point",
            "properties": OrderedDict([("sinuosity", "float")]),
        }

        with fiona.open(
            sinuosity_vector,
            "w",
            driver="GPKG",
            crs=vect.crs,
            schema=output_schema,
        ) as out_vect:
            for point in points:
                out_vect.write({
                    "geometry": {
                        "type": "Point",
                        "coordinates": (point[0], point[1]),
                    },
                    "properties": OrderedDict([("sinuosity", point[2])]),
                })


def stream_slope(
    dem: str,
    streams: str,
    slope_dst: str,
    units: str = "degrees",
    scale: float = 1,
    focal_mean_dist: None | float = None,
):
    """Calculate slope along extracted streams.

    Args:
        dem (str): Digital Elevation model raster
        streams (str): Streams raster source generated using `watershed.extract_streams`
        slope_dst (str): Destination slope raster dataset.
        units (str, optional): Units for the output. Defaults to "degrees".
        scale (float, optional): Z-factor to scale the output. Defaults to 1.
        focal_mean_dist (Union[None, float]): Sampling distance for a focal mean
        designed to smooth slope values so they are representative of segments and
        avoid oversampling at the cell level. Defaults to None.
    """
    with TempRasterFile() as elev_dst:
        to_raster(
            da.ma.masked_where(
                da.ma.getmaskarray(from_raster(streams)), from_raster(dem)
            ),
            dem,
            elev_dst,
            as_cog=False,
        )

        if focal_mean_dist is not None:
            with TempRasterFile() as slope_dst_tmp:
                slope(elev_dst, slope_dst_tmp, units, scale)

                raster_specs = Raster.raster_specs(dem)

                raster_filter(
                    slope_dst_tmp,
                    kernel_from_distance(
                        float(focal_mean_dist), raster_specs["csx"], raster_specs["csy"]
                    ),
                    "mean",
                    slope_dst,
                )
        else:
            slope(elev_dst, slope_dst, units, scale)


def upstream_stats(
    values: str,
    streams: str,
    flow_direction: str,
    gradient_dst: str,
    distance: float,
):
    """Compute a moving mean of any stream-aligned raster over a distance upstream.

    All first-order stream inlets (cells with no upstream stream neighbours) are
    identified.  From each inlet the stream network is traversed downstream.  A
    sliding window of ``distance`` map units is maintained: the value at each
    visited cell is added to the back of the window, and entries are dropped from
    the front once they are more than ``distance`` upstream of the current cell.
    The mean of the values in the window is recorded at each cell visited.

    The ``values`` raster can be anything co-registered with the stream network —
    a pre-computed slope (via ``stream_slope``), channel width, precipitation, etc.
    Cells downstream of confluences may be visited by more than one inlet traversal;
    the last write prevails.

    Args:
        values (str): Raster whose values are to be averaged upstream. Must be
            co-registered with ``streams`` and cover all stream cell locations.
        streams (str): Streams raster derived from ``watershed.extract_streams``.
        flow_direction (str): Flow direction raster from ``watershed.extract_streams``.
        gradient_dst (str): Output raster path.
        distance (float): Sliding window length in CRS linear units (e.g. metres).
    """
    raster_specs = Raster.raster_specs(streams)
    csx = float(raster_specs["csx"])
    csy = float(raster_specs["csy"])

    # Row/col offsets indexed by fd value 0-8 (0 is unused)
    fd_off_i = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int32)
    fd_off_j = np.array([0, 1, 0, -1, -1, -1, 0, 1, 1], dtype=np.int32)

    # +2: one for the inlet seed, one buffer slot so tail never catches head
    max_window = int(np.ceil(distance / min(csx, csy))) + 2

    streams_dask = from_raster(streams)
    values_arr = np.squeeze(da.ma.filled(from_raster(values), np.nan).compute()).astype(
        np.float64
    )
    streams_mask = np.squeeze(~da.ma.getmaskarray(streams_dask).compute())
    fd_arr = np.squeeze(da.ma.filled(from_raster(flow_direction), 0).compute()).astype(
        np.int32
    )

    n_rows, n_cols = values_arr.shape
    gradient_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    gradient_arr = _upstream_compute(
        values_arr,
        streams_mask,
        fd_arr,
        gradient_arr,
        fd_off_i,
        fd_off_j,
        csx,
        csy,
        float(distance),
        max_window,
    )

    out = da.from_array(
        gradient_arr.reshape(streams_dask.shape),
        chunks=streams_dask.chunks,
    )
    to_raster(da.ma.masked_invalid(out), streams, gradient_dst)


def bankfull_width_geometric(
    streams: str,
    slope: str,
    bankfull_dst: str,
    lcp_dst: str = None,
    method: str = "balltree",
    max_cost: float = 0.8,
    memory: int = None,
    input_lcp=None,
):
    """Calculate bankfull width as a product of the valley geometry. Slope is used as
    cost for a Least Cost Path surface originating at streams. A threshold cost is
    provided to constrain regions that depict potential bankfull. The resulting distance
    across these regions is applied to stream locations as bankfull width.

    Args:
        streams (str): Stream raster derived from `watershed.extract_streams`
        slope (str): Slope raster in degrees derived from `elevation.slope`
        bankfull_dst (str): Output Bankfull Width raster
        lcp_dst (str, optional): Output raster with the Least Cost Path surface.
        Defaults to None.
        max_cost (float, optional): Maximum cost to constrain regions of bankfull
        extent. Defaults to 0.13.
        memory (int, optional): Maximum memory usage for GRASS operations.
        Defaults to None.
    """

    @njit
    def filled_dfe(edges, dfe, dfe_order, cs):
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

        shape = edges.shape

        width = np.zeros_like(dfe)

        for o_i in range(dfe_order.shape[0]):
            r_i, r_j = dfe_order[o_i]

            if dfe[r_i, r_j] == -1:
                continue

            width_val = dfe[r_i, r_j] * 2 if dfe[r_i, r_j] > 0 else cs

            width[r_i, r_j] = width_val
            dfe[r_j, r_j] = -1

            i = r_i
            j = r_j
            while True:
                edge_encounter = False
                min_i, min_j = -1, -1
                _m = np.inf
                for i_off, j_off in offsets:
                    i_nbr = i + i_off
                    j_nbr = j + j_off
                    if (
                        i_nbr == shape[0]
                        or i_nbr < 0
                        or j_nbr == shape[1]
                        or j_nbr < 0
                        or dfe[i_nbr, j_nbr] == -1
                    ):
                        continue

                    if dfe[i_nbr, j_nbr] < _m:
                        _m = dfe[i_nbr, j_nbr]
                        min_i, min_j = i_nbr, j_nbr

                    if edges[i_nbr, j_nbr]:
                        edge_encounter = True

                if min_i == -1 or min_j == -1:
                    break

                dfe[min_i, min_j] = -1
                width[min_i, min_j] = width[r_i, r_j]
                i, j = min_i, min_j

                if edge_encounter:
                    break

        return width

    raster_specs = Raster.raster_specs(slope)
    avg_cs = (raster_specs["csx"] + raster_specs["csy"]) / 2.0

    with TempRasterFiles(4) as (cost_path, lcp_path, edges_src, dt_from_edges_src):
        if input_lcp is None:
            # Generate cost surface (normalized slope, adjusted for cell size)
            cost = (
                da.clip(from_raster(slope).astype(np.float64), 0.0, 90.0) / 90.0
            ) * avg_cs

            # Override temporary LCP path if one is provided
            if lcp_dst is not None:
                lcp_path = lcp_dst

            # Generate least cost surface and closest stream map
            to_raster(cost, slope, cost_path, as_cog=False)

            cmd_kwargs = {
                "input": "cost",
                "output": "cost_path",
                "start_raster": "streams",
                "flags": "k",
            }

            if memory is not None:
                cmd_kwargs["memory"] = memory

            with GrassRunner(cost_path) as gr:
                gr.run_command(
                    "r.cost",
                    (cost_path, "cost", "raster"),
                    (streams, "streams", "raster"),
                    **cmd_kwargs,
                )
                gr.save_raster("cost_path", lcp_path, as_cog=lcp_dst is not None)
        else:
            lcp_path = input_lcp

        region = from_raster(lcp_path) < max_cost

        # Find the edges
        eroded = binary_erosion(region, np.ones((1, 3, 3), bool))
        edges = region & ~eroded

        if method == "growfill":
            # Distance transform to edges
            to_raster(edges, slope, edges_src, as_cog=False)

            distance_transform(edges_src, dt_from_edges_src)

            dfe = from_raster(dt_from_edges_src)

            dfe = da.ma.filled(da.where(region, dfe, -1), -1).compute()[0, ...]

            dfe_order = np.unravel_index(np.argsort(dfe.ravel())[::-1], dfe.shape)

            width = filled_dfe(
                da.ma.filled(edges, False).compute()[0, ...],
                dfe,
                np.array(dfe_order).T,
                avg_cs,
            )

            width = da.from_array(width.reshape(edges.shape), chunks=edges.chunks)
            width = da.ma.masked_where(width == 0, width)

            to_raster(
                width,
                lcp_path,
                bankfull_dst,
            )
        elif method == "balltree":
            region_width(region, streams, bankfull_dst)


def region_width(
    region: str | da.Array,
    streams: str,
    width_dst: str,
):
    """Measure the width of a region surrounding streams, attributed to stream cells.

    Region edge cells are assigned to their nearest stream cell and the mean distance
    from each stream cell to its edge cells is doubled to represent the full width
    across the stream. Stream cells without edge cells are interpolated from their
    neighbours.

    Args:
        region (str | da.Array): Boolean region raster (nodata or 0 outside), or a
            boolean dask array in raster shape.
        streams (str): Stream raster derived from `watershed.extract_streams`.
        width_dst (str): Output width raster (m) on stream cells.
    """
    raster_specs = Raster.raster_specs(streams)
    avg_cs = (raster_specs["csx"] + raster_specs["csy"]) / 2

    if isinstance(region, str):
        region_a = from_raster(region)
        region = ~da.ma.getmaskarray(region_a) & (da.ma.filled(region_a, 0) != 0)

    eroded = binary_erosion(region, np.ones((1, 3, 3), bool))
    edges = region & ~eroded

    # Accumulate the distance to edges along streams
    edge_points = np.array(da.where(edges[0])).T.astype("float32")
    edge_points[:, 0] *= raster_specs["csy"]
    edge_points[:, 1] *= raster_specs["csx"]

    stream_mask = ~da.ma.getmaskarray(from_raster(streams))

    stream_inds = np.array(da.where(stream_mask[0])).T
    stream_points = stream_inds.astype("float32")
    stream_points[:, 0] *= raster_specs["csy"]
    stream_points[:, 1] *= raster_specs["csx"]

    bt = BallTree(stream_points)
    distances, indices = map(lambda x: x.ravel(), bt.query(edge_points, 1))

    bins = np.bincount(indices, distances)
    indices, counts = np.unique(indices, return_counts=True)
    # Distance is the average of distance to surrounding banks * 2 to account for
    # the entire valley width.
    distances = (bins[indices] / counts) * 2

    width = np.full(stream_mask.shape[1:], -1, np.float32)
    width[(stream_inds[indices][:, 0], stream_inds[indices][:, 1])] = distances
    width[width == 0] = avg_cs

    # Interpolate where streams are missing values
    valid = width != -1
    points = np.array(np.where(valid)).T
    values = width[valid]
    xi = np.array(np.where(~valid & stream_mask.compute()[0])).T

    if len(xi) > 0:
        width[(xi[:, 0], xi[:, 1])] = griddata(points, values, xi, fill_value=avg_cs)

    out_dask = da.from_array(width.reshape(stream_mask.shape)).rechunk(
        stream_mask.chunks
    )

    to_raster(
        da.ma.masked_where(out_dask == -1, out_dask),
        streams,
        width_dst,
    )


def fit_morphology_equation(
    csv_data: str, obs_field: str, **kwargs
) -> tuple[float, float, float]:
    """
    Optimize the coeff, a_exp, and p_exp parameters in the stream morphology equation:

    coeff * (ca_a**a_exp) * (precip_a**p_exp)

    Where ca_a is the contributing area in km**2 and precip_a is the mean annual
    precipitation in mm.

    Args:
        csv_data (str): Path to a CSV file containing columns "ca", and "precip" for
        contributing area (in km**2), mean annual precipitation (in mm). The observed
        field is specified using the `obs_field` argument.
        obs_field (str): The name of the column in the CSV file that contains the
        observed data to fit.

    Kwargs:
        performance_plot (bool): Whether to generate a performance plot comparing
        observed and predicted values. Defaults to False.

    Returns:
        tuple: Fitted parameters (coeff, a_exp, p_exp).

    """

    def bfw_model(params, ca, precip):
        bw_coeff, a_exp, p_exp = params
        return bw_coeff * (ca**a_exp) * (precip**p_exp)

    def objective_function(params, data):
        predicted_bfw = bfw_model(params, data["ca"], data["precip"] / 10)
        residuals = data[obs_field] - predicted_bfw
        return np.sum(residuals**2)

    def plot_performance(fitted_params, data, obs_field):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warning(
                "matplotlib is not installed. Performance plot cannot be generated."
            )
            return

        predicted_bfw = bfw_model(fitted_params, data["ca"], data["precip"] / 10)

        # Calculate the model error and r2 to show on the plot
        mse = mean_absolute_error(data[obs_field], predicted_bfw)
        r2 = r2_score(data[obs_field], predicted_bfw)

        plt.scatter(data[obs_field], predicted_bfw)
        plt.plot(
            [data[obs_field].min(), data[obs_field].max()],
            [data[obs_field].min(), data[obs_field].max()],
            "r--",
            label="1:1 Line",
        )
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title("Morphologic Parameter model performance")
        plt.legend()

        plt.text(
            0.05,
            0.95,
            f"MAE: {mse:.2f}\nR²: {r2:.2f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )

        plt.gca().set_aspect("equal", adjustable="box")

        plt.show()

    data = np.genfromtxt(csv_data, delimiter=",", names=True)

    if (include_field := kwargs.get("include_field", None)) is not None:
        data = data[data[include_field] == 1]

    initial_params = [0.04, 0.5, 0.7]  # Initial guesses

    result = minimize(
        objective_function,
        initial_params,
        args=(data,),
        method=kwargs.get("method", "Nelder-Mead"),
    )

    if result.success:
        fitted_params = result.x

        print(
            f"Fitted parameters: coeff={fitted_params[0]}, a_exp={fitted_params[1]}, p_exp={fitted_params[2]}"
        )

        if kwargs.get("performance_plot", False):
            plot_performance(fitted_params, data, obs_field)

        return fitted_params
    else:
        raise RuntimeError("Optimization failed to converge.")


def valley_width(
    streams: str,
    slope: str,
    dst: str,
    lcp_dst: str = None,
    max_cost: float = 1,
    memory: int = None,
):
    """Mirror of bankfull_width_geometric"""
    return bankfull_width_geometric(
        streams, slope, dst, lcp_dst=lcp_dst, max_cost=max_cost, memory=memory
    )


def bankfull_width(
    streams: str,
    flow_direction: str,
    precip: str,
    dst: str,
    bw_coeff: float = 0.042,
    a_exp: float = 0.48,
    p_exp: float = 0.74,
):
    """Estimate bankfull width with empirically-derived constants using:

    bankfull_width = bw_coeff * \
        (contrib_area[km2] ** a_exp) * (precip[cm/yr] ** p_exp)

    Method and defaults from:

    Hall, J. E., D. M. Holzer, and T. J. Beechie. 2007. Predicting river floodplain and
        lateral channel migration for salmon habitat conservation. Journal of the
        American Water Resources Association 43:786-797.

    Note, ensure the spatial reference of the rasters is a projected system in metres.

    An example workflow that starts with a DEM and uses streams with a minimum
    contributing area could look like this:

    ```python
    with TempRasterFile() as tmp_flow_direction:
        # Extract streams
        flow_direction_accumulation(
            dem_path,
            tmp_flow_direction,
            accumulation_dst
        )

        # Calculate flow direction and streams (example uses a 1km2 threshold)
        extract_streams(
            dem_path, accumulation_dst, stream_dst, flow_direction_dst, 1e6, 0
        )

        # Calculate bankfull width
        bankfull_width(
            stream_dst,
            flow_direction_dst,
            avg_annual_precip,
            bankfull_dst,
        )
    ```

    Args:
        streams: Grid with stream locations
        (generated using `watershed.extract_streams`),
        direction: Grid with flow direction
        (generated using `watershed.extract_streams`),
        precip (str): Mean annual precipitation grid in mm
        dst (str): Destination raster for bankfull grid
        bw_coeff (float): Bankfull width coefficient. Defaults to 0.196.
        a_exp (float): Area expenential scale. Defaults to 0.280.
        p_exp (float): Precip expenential scale. Defaults to 0.355.
    """
    # Calculate contributing area and mean annual precip
    with TempRasterFiles(3) as (ca, mean_precip, precip_aligned):
        if not Raster(precip).matches(flow_direction):
            warp_like(
                precip,
                precip_aligned,
                flow_direction,
                resample_method="bilinear",
                as_cog=False,
            )
            precip = precip_aligned

        fd = FlowAccumulation(flow_direction)
        fd.contributing_area(ca)
        fd.calculate(precip, mean_precip, "mean")

        precip_a = from_raster(mean_precip) / 10  # mm -> cm
        ca_a = from_raster(ca)

        bankfull_width = bw_coeff * (ca_a**a_exp) * (precip_a**p_exp)

        # Isolate streams
        streams_a = from_raster(streams)

        to_raster(
            da.ma.masked_where(da.ma.getmaskarray(streams_a), bankfull_width),
            streams,
            dst,
        )


def bankfull_depth(
    bankfull_width_src: str,
    bankfull_depth_dst: str,
    b_d_coeff: float = 0.145,
    b_w_exp: float = 0.607,
):
    """Generate a bankfull depth dataset. Method and defaults from:

    Hall, J. E., D. M. Holzer, and T. J. Beechie. 2007. Predicting river floodplain and
        lateral channel migration for salmon habitat conservation. Journal of the
        American Water Resources Association 43:786-797.

    Args:
        bankfull_width_src (str): Bankfull width along streams derived from
        `morphology.bankfull_width`.
        bankfull_depth_dst (str): Output bankfull depth or bankfull mask dataset.
        b_d_coeff (float, optional): Bankfull depth coefficient. Defaults to 0.145.
        b_w_exp (float, optional): Bankfull width exponent. Defaults to 0.607.
    """
    bankfull_depth = b_d_coeff * (from_raster(bankfull_width_src) ** b_w_exp)

    to_raster(
        bankfull_depth,
        bankfull_width_src,
        bankfull_depth_dst,
    )


def bankfull_extent(
    bankfull_width_src: str,
    bankfull_depth_src: str,
    dem: str,
    bankfull_extent_dst: str,
    flood_factor: float = 1,
):
    """Interpolate a bankfull extent surrounding streams using the bankfull elevation.

    **Note**: This method is not memory-safe.

    Args:
        bankfull_width_src (str): Bankfull Depth calculated using
        `morphology.bankfull_width`.
        bankfull_depth_src (str): Bankfull Depth calculated using
        `morphology.bankfull_depth`.
        dem (str): DEM Grid used to originally derive Bankfull Depth.
        bankfull_extent_dst (str): Output raster with the bankfull extent mask.
        flood_factor (float, optional): A scaling factor to modify the bankfull depth.
        Defaults to 1.
    """
    elevation = from_raster(dem)
    bf_elevation = elevation + from_raster(bankfull_depth_src) * flood_factor

    # Load stream locations and DEM into memory
    (_, i, j), complete, bfe, bfw, str_dist, elev = da.compute(
        da.where(~da.ma.getmaskarray(bf_elevation)),
        ~da.ma.getmaskarray(bf_elevation),
        da.ma.filled(bf_elevation, -999),
        from_raster(bankfull_width_src),
        da.zeros_like(elevation, dtype="float32"),
        da.ma.filled(elevation, -999),
    )

    raster_specs = Raster.raster_specs(dem)
    shape = raster_specs["shape"][1], raster_specs["shape"][2]
    csx, csy = raster_specs["csx"], raster_specs["csy"]

    stack = List(np.array([i, j]).T.tolist())
    del i, j

    @njit(parallel=True)
    def interpolate(
        stack,
        complete,
        bfe,
        bfw,
        str_dist,
        elev,
        i_bound,
        j_bound,
        i_sampling,
        j_sampling,
    ):
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
                    if i_nbr == i_bound or i_nbr < 0 or j_nbr == j_bound or j_nbr < 0:
                        continue

                    if (
                        elev[0, i_nbr, j_nbr] != -999
                        and bfe[0, i_nbr, j_nbr] == -999
                        and not complete[0, i_nbr, j_nbr]
                    ):
                        stack_2.append([i_nbr, j_nbr])
                        complete[0, i_nbr, j_nbr] = True

            while len(stack_2) > 0:
                i, j = stack_2.pop()
                bfe_accum = 0
                bfw_accum = 0
                dist_accum = 0
                modal = 0.0
                distance = 0
                inverse_distance = 0
                for i_off, j_off in offsets:
                    i_nbr = i + i_off
                    j_nbr = j + j_off

                    if i_nbr == i_bound or i_nbr < 0 or j_nbr == j_bound or j_nbr < 0:
                        continue

                    dist = np.sqrt(
                        (i - i_nbr * i_sampling) ** 2.0
                        + (j - j_nbr * j_sampling) ** 2.0
                    )
                    inverse_dist = 1 / dist

                    if bfe[0, i_nbr, j_nbr] != -999:
                        bfe_accum += bfe[0, i_nbr, j_nbr] * inverse_dist
                        bfw_accum += bfw[0, i_nbr, j_nbr] * inverse_dist
                        dist_accum += str_dist[0, i_nbr, j_nbr] * inverse_dist
                        inverse_distance += inverse_dist

                        distance += dist
                        modal += 1

                bfe_accum /= inverse_distance
                bfw_accum /= inverse_distance
                dist_accum /= inverse_distance

                avg_width = distance / modal
                dist_accum += avg_width
                limit = elev[0, i, j]

                if bfe_accum >= limit and dist_accum <= bfw_accum:
                    stack.append([i, j])
                    bfe[0, i, j] = bfe_accum
                    bfw[0, i, j] = bfw_accum
                    str_dist[0, i, j] = dist_accum

    interpolate(stack, complete, bfe, bfw, str_dist, elev, shape[0], shape[1], csy, csx)

    to_raster(
        da.from_array(bfe != -999),
        bankfull_depth_src,
        bankfull_extent_dst,
    )


def valley_confinement(
    bankfull_width_src: str,
    valley_width_src: str,
    valley_confinement_dst: str,
    max_width: float = 1000,
):
    """Calculate a valley confinement index - the ratio of valley width to estimated
    bankfull width.

    Args:
        bankfull_width_src (str): Bankfull width data source calculated using
        `morphology.bankfull_width`.
        valley_width (str): Valley width calculated using `morphology.valley_width`.
        valley_confinement_dst (str): Output raster with valley confinement index
        values.
    """
    bfw = from_raster(bankfull_width_src)
    valleys = from_raster(valley_width_src)

    vc = da.clip(valleys, 0, max_width) / da.ma.masked_where(bfw <= 0, bfw)

    to_raster(
        vc,
        bankfull_width_src,
        valley_confinement_dst,
    )


def topographic_wetness(
    streams: str,
    slope_src: str,
    topographic_wetness_dst: str,
    max_cost: float = 750.0,
    knights_move: bool = True,
    memory=4096,
):
    """Calculate a topographic wetness index

    Args:
        streams (str): Grid with stream locations
        (generated using `watershed.flow_direction_accumulation`)
        slope_src (str): Topographic slope in Degrees.
        topographic_wetness_dst (str): Output TWI grid
        max_cost (float): Maximum cost used to scale and invert the TWI values.
        Defaults to 750.
        memory (int, optional): Maximum memory usage for GRASS operations.
    """
    raster_specs = Raster.raster_specs(slope_src)
    avg_cs = (raster_specs["csx"] + raster_specs["csy"]) / 2.0

    streams = ~da.ma.getmaskarray(from_raster(streams))

    # Generate cost surface (normalized slope, adjusted for cell size)
    cost = (
        da.clip(from_raster(slope_src).astype(np.float64), 0.0, 90.0) / 90.0
    ) * avg_cs
    # Make cost 0 where simulated streams exist
    cost = raster_where(~streams, cost, 0)

    # Generate least cost surface
    with TempRasterFiles(3) as (streams_path, cost_path, cost_surface):
        to_raster(cost, slope_src, cost_path, as_cog=False)
        to_raster(streams, slope_src, streams_path, as_cog=False)

        kwargs = {
            "input": "cost",
            "output": "cost_path",
            "start_raster": "streams",
        }

        if knights_move:
            kwargs["flags"] = "k"

        if memory is not None:
            kwargs["memory"] = memory

        with GrassRunner(cost_path) as gr:
            gr.run_command(
                "r.cost",
                (cost_path, "cost", "raster"),
                (streams_path, "streams", "raster"),
                **kwargs,
            )
            gr.save_raster("cost_path", cost_surface)

        cs = from_raster(cost_surface).astype("float32")

        # Invert
        cs_inverted = max_cost - da.ma.masked_where(cs > max_cost, cs)

        to_raster(cs_inverted, slope_src, topographic_wetness_dst)


def slope_modifier(
    stream_slope: da.Array,
    slope_breaks: tuple[float, float] = (0.015, 0.04),
    g_range: tuple[float, float] = (0.5, 1.25),
) -> da.Array:
    """Map channel gradient to a floodplain-width modifier.

    Low-gradient (pool-riffle) reaches build floodplains and receive `g_max`;
    steep (step-pool/cascade) reaches receive `g_min`; the modifier varies linearly
    between the two gradient breaks. Breaks follow the process-domain thresholds of
    Montgomery & Buffington (1997).

    Args:
        stream_slope (da.Array): Channel gradient as a fraction (m/m).
        slope_breaks (tuple[float, float], optional): Gradients (m/m) at which the
            modifier reaches `g_max` and `g_min` respectively.
            Defaults to (0.015, 0.04).
        g_range (tuple[float, float], optional): (`g_min`, `g_max`) bounds of the
            modifier. Defaults to (0.5, 1.25).

    Returns:
        da.Array: Modifier in the range [`g_min`, `g_max`].
    """
    s_low, s_high = float(slope_breaks[0]), float(slope_breaks[1])
    g_min, g_max = float(g_range[0]), float(g_range[1])

    if s_high <= s_low:
        raise ValueError("slope_breaks must be increasing.")
    if g_min <= 0 or g_max < g_min:
        raise ValueError("g_range must be positive and ordered (g_min, g_max).")

    frac = da.clip((stream_slope - s_low) / (s_high - s_low), 0.0, 1.0)

    return g_max - frac * (g_max - g_min)


def rise_cost_surface(
    slope_src: str,
    start_cost_src: str,
    cost_dst: str,
    nearest_dst: str,
    cost_floor: float = DEFAULT_COST_FLOOR,
    max_cost: float | None = None,
    weight_src: str | None = None,
    knights_move: bool = True,
    memory: int | None = 4096,
):
    """Accumulate topographic rise away from streams using a least cost path.

    The friction surface is `cell size * (tan(slope) + cost_floor)`, so the
    cumulative cost approximates the metres of climb along the least-climb path from
    the stream (a path-based height above nearest drainage), plus a small distance
    term that bounds the lateral extent across flat terrain.

    Each stream cell is seeded with the value in `start_cost_src` (GRASS `r.cost -r`),
    which allows a per-stream budget: seeding with `T_max - T_i` and thresholding the
    output at `T_max` is equivalent to allowing stream `i` to spend `T_i` metres of
    rise. Channel cells carry the same friction as any other cell, so the budget of a
    larger stream extends up a tributary only until the channel has climbed the
    difference in budgets (a backwater limit), beyond which the tributary's own seed
    prevails. The `nearest_dst` output holds the seed value of the stream that reached
    each cell at least cost.

    Args:
        slope_src (str): Topographic slope in degrees.
        start_cost_src (str): Raster with seed costs on stream cells and nodata
            elsewhere.
        cost_dst (str): Output cumulative cost raster.
        nearest_dst (str): Output raster with the seed value of the nearest stream.
        cost_floor (float, optional): Minimum friction expressed as a gradient (m/m).
            Defaults to tan(0.5 degrees).
        max_cost (float | None, optional): Stop accumulating beyond this cumulative
            cost; cells beyond it are nodata. Defaults to None (unbounded).
        weight_src (str | None, optional): Raster of multiplicative friction weights,
            for example from `convergence_weight`. Nodata weights are treated as 1.
            Defaults to None.
        knights_move (bool, optional): Use the knight's move in `r.cost`.
            Defaults to True.
        memory (int | None, optional): Memory (MB) for `r.cost`. Defaults to 4096.
    """
    if cost_floor <= 0:
        raise ValueError("cost_floor must be positive to bound extent on flat terrain.")

    raster_specs = Raster.raster_specs(slope_src)
    avg_cs = (raster_specs["csx"] + raster_specs["csy"]) / 2.0

    slope_deg = da.clip(from_raster(slope_src).astype(np.float64), 0.0, 89.0)
    cost = (da.tan(da.radians(slope_deg)) + cost_floor) * avg_cs

    if weight_src is not None:
        weight = da.ma.filled(from_raster(weight_src).astype(np.float64), 1.0)
        cost = cost * weight

    with TempRasterFile() as cost_path:
        to_raster(cost, slope_src, cost_path, as_cog=False)

        kwargs = {
            "input": "cost",
            "output": "cum_cost",
            "start_raster": "start_cost",
            "nearest": "nearest",
            "flags": "kr" if knights_move else "r",
        }

        if memory is not None:
            kwargs["memory"] = memory

        # r.cost accepts an integer maximum; 0 disables the limit
        if max_cost is not None and max_cost > 0:
            kwargs["max_cost"] = int(math.ceil(max_cost)) + 1

        with GrassRunner(cost_path) as gr:
            gr.run_command(
                "r.cost",
                (cost_path, "cost", "raster"),
                (start_cost_src, "start_cost", "raster"),
                **kwargs,
            )
            gr.save_raster("cum_cost", cost_dst)
            gr.save_raster("nearest", nearest_dst)


def riparian_zones(
    cost_src: str,
    nearest_src: str,
    t_max: float,
    f1: float,
    f2: float,
    h3: float,
    zones_dst: str,
    score_dst: str | None = None,
    streams: str | None = None,
):
    """Classify a seeded rise-cost surface into three nested riparian zones.

    With seeds of `T_max - T3_i`, the budget of the stream that reached a cell is
    recovered as `T3 = t_max - nearest`, the rise actually spent as
    `rise = cost - nearest`, and the inner budgets as `T2 = T3 - h3` and
    `T1 = T2 * f1 / f2`.

    Args:
        cost_src (str): Cumulative cost from `rise_cost_surface`.
        nearest_src (str): Nearest-seed raster from `rise_cost_surface`.
        t_max (float): Maximum zone 3 budget used to construct the seeds.
        f1 (float): Bankfull depth multiple defining zone 1.
        f2 (float): Bankfull depth multiple defining zone 2.
        h3 (float): Rise (m) above the zone 2 limit defining zone 3.
        zones_dst (str): Output zone raster (1, 2, 3; 0 is nodata).
        score_dst (str | None, optional): Output raster with the fraction of the
            total budget remaining at each cell (1 at streams, 0 at the outer edge).
            Defaults to None.
        streams (str | None, optional): Stream raster. Channel cells are always
            zone 1 with a score of 1, even where a neighbouring stream's seed reached
            them more cheaply than their own. Defaults to None.
    """
    cost = from_raster(cost_src).astype(np.float64)
    c0 = from_raster(nearest_src).astype(np.float64)

    rise = cost - c0
    t3 = float(t_max) - c0
    t2 = t3 - float(h3)
    t1 = t2 * (float(f1) / float(f2))

    zones = da.where(rise < t1, 1, da.where(rise < t2, 2, da.where(rise < t3, 3, 0)))
    score = da.clip((t3 - rise) / t3, 0.0, 1.0)

    mask = da.ma.getmaskarray(cost) | da.ma.getmaskarray(c0) | (zones == 0)

    if streams is not None:
        stream_mask = ~da.ma.getmaskarray(from_raster(streams))
        zones = da.where(stream_mask, 1, zones)
        score = da.where(stream_mask, 1.0, score)
        mask = mask & ~stream_mask

    to_raster(
        da.ma.masked_where(mask, zones.astype(np.uint8)),
        cost_src,
        zones_dst,
        nodata_value=0,
    )

    if score_dst is not None:
        to_raster(da.ma.masked_where(mask, score.astype(np.float32)), cost_src, score_dst)


def convergence_weight(
    flow_accumulation: da.Array,
    cell_area: float,
    a_hill: float,
    gamma: float = 0.5,
    w_min: float = 0.25,
) -> da.Array:
    """Friction weight that lowers resistance in convergent terrain.

    Hillslope groundwater discharges into hollows and toe slopes in proportion to the
    area draining to them, so convergent cells remain connected to the riparian water
    table at greater rise than planar or divergent cells. The weight is

        w = clip((a_hill / a) ** gamma, w_min, 1)

    where `a` is contributing area. Cells draining less than `a_hill` are unaffected.

    Args:
        flow_accumulation (da.Array): Contributing cell count from
            `watershed.flow_direction_accumulation`.
        cell_area (float): Area of a cell (m2) used to convert counts to area.
        a_hill (float): Contributing area (m2) above which terrain is considered
            convergent.
        gamma (float, optional): Rate at which friction decreases with area.
            Defaults to 0.5.
        w_min (float, optional): Minimum weight. Defaults to 0.25.

    Returns:
        da.Array: Weights in the range [`w_min`, 1].
    """
    if a_hill <= 0 or gamma <= 0 or not 0 < w_min <= 1:
        raise ValueError("a_hill and gamma must be positive and 0 < w_min <= 1.")

    area = da.maximum(flow_accumulation.astype(np.float64), 1.0) * float(cell_area)

    return da.clip((float(a_hill) / area) ** float(gamma), float(w_min), 1.0)


def extend_fringe(
    fringe_cost_src: str,
    fringe_nearest_src: str,
    t_max: float,
    h3: float,
    zones_src: str,
    score_src: str,
    zones_dst: str,
    score_dst: str,
):
    """Union a second-pass fringe with existing zones.

    The second pass is seeded from zone 1 and 2 cells with their first-pass seed
    values, so `t_max - nearest` recovers the stream budget and `cost - nearest` the
    friction spent from the floodplain edge. Cells within `h3` of the edge that are not
    already in zones 1 or 2 become zone 3.

    Args:
        fringe_cost_src (str): Cumulative cost from the second `rise_cost_surface` pass.
        fringe_nearest_src (str): Nearest-seed raster from the second pass.
        t_max (float): Maximum zone 3 budget used to construct the first-pass seeds.
        h3 (float): Fringe budget.
        zones_src (str): First-pass zones from `riparian_zones`.
        score_src (str): First-pass score from `riparian_zones`.
        zones_dst (str): Output zones.
        score_dst (str): Output score.
    """
    zones = from_raster(zones_src)
    score = from_raster(score_src).astype(np.float64)

    cost = from_raster(fringe_cost_src).astype(np.float64)
    c0 = from_raster(fringe_nearest_src).astype(np.float64)
    spent = cost - c0
    t3 = float(t_max) - c0

    floodplain = ~da.ma.getmaskarray(zones) & (zones <= 2)
    reached = ~da.ma.getmaskarray(cost) & (spent < float(h3)) & ~floodplain

    fringe_score = da.clip((float(h3) - spent) / t3, 0.0, 1.0)

    zones_f = da.ma.filled(zones, 0)
    out_zones = da.where(reached & (zones_f == 0), 3, zones_f)
    out_score = da.where(
        reached, da.maximum(da.ma.filled(score, 0.0), fringe_score), da.ma.filled(score, 0.0)
    )

    mask = out_zones == 0

    to_raster(
        da.ma.masked_where(mask, out_zones.astype(np.uint8)),
        zones_src,
        zones_dst,
        nodata_value=0,
    )
    to_raster(da.ma.masked_where(mask, out_score.astype(np.float32)), zones_src, score_dst)


class RiparianConnectivity:
    """Riparian zone delineation workflow

    Three nested zones are delineated around a stream network by accumulating
    topographic rise (a path-based height above the stream) away from each stream cell
    and comparing it with per-stream budgets:

    * Zone 1 - channel and active floodplain: rise < `f1 * D_bf * g`
    * Zone 2 - flood-prone area: rise < `f2 * D_bf * g`
    * Zone 3 - groundwater-connected fringe: rise < zone 2 limit + `h3`

    where `D_bf` is bankfull depth derived from bankfull width (contributing area and
    mean annual precipitation; Hall et al. 2007), `g` is a channel-gradient modifier
    (low-gradient reaches build wider floodplains), and `h3` is an absolute band in
    metres representing a water table that follows topography and remains within
    rooting depth of the surface.

    The general steps are:

    1. Instantiate using a DEM and a mean annual precipitation grid (mm)

        `rc = RiparianConnectivity(dem_path, precip_path)`

    2. Extract streams

        `rc.extract_streams()`

    3. Calculate stream attributes (bankfull width and depth, channel gradient)

        `rc.calc_stream_attributes()`

    4. Delineate zones

        `rc.calc_zones()`

        Inspect the `zones` raster and adjust `f1`, `f2`, `h3` or the gradient
        modifier as needed, then recalculate.

    5. Save the result

        `rc.calc_connectivity(output_dataset)`
    """

    _attributes = (
        "slope",
        "flow_direction",
        "flow_accumulation",
        "streams",
        "bankfull_width",
        "bankfull_depth",
        "stream_slope",
        "start_cost",
        "cost",
        "nearest",
        "fringe_cost",
        "fringe_nearest",
        "zones",
        "score",
        "region",
        "t_max",
        "min_ws_area",
    )

    def __init__(self, dem: str, annual_precip: str):
        """Define a study area and begin a riparian analysis.

        Args:
            dem (str): Digital Elevation Model raster in a projected system (metres).
            annual_precip (str): Mean annual precipitation grid in mm. The grid is
                warped to the DEM if it does not align.
        """
        self.dem = dem

        # Specifications
        self.__dict__.update(Raster.raster_specs(dem))

        # Create a directory
        self.tmp_dir = TemporaryDirectory(dir=TMP_DIR)

        for attr in self._attributes:
            setattr(self, attr, None)

        self.precip = self._prepare_precip(annual_precip)

    def _prepare_precip(self, annual_precip: str) -> str:
        if Raster(annual_precip).matches(self.dem):
            return annual_precip

        precip_dst = self.raster_path("precip")
        warp_like(
            annual_precip,
            precip_dst,
            self.dem,
            resample_method="bilinear",
            as_cog=False,
        )
        return precip_dst

    def raster_path(self, name: str) -> str:
        """Generate a raster path using the current temporary directory.

        Args:
            name (str): Name of the raster dataset.

        Returns:
            str: A valid raster path.
        """
        return os.path.join(self.tmp_dir.name, f"{name}.tif")

    def verify_source(self, src: str):
        """Ensure a raster aligns with the current study area.

        Args:
            src (str): Source Raster dataset.
        """
        src_specs = Raster(src)
        if not all([
            np.isclose(src_specs.csx, self.csx),
            np.isclose(src_specs.csy, self.csy),
            np.isclose(src_specs.top, self.top),
            np.isclose(src_specs.bottom, self.bottom),
            np.isclose(src_specs.left, self.left),
            np.isclose(src_specs.right, self.right),
            src_specs.shape == self.shape,
            compare_projections(src_specs.wkt, self.wkt),
        ]):
            raise ValueError(f"Input raster {src} does not align with the study area.")

    def extract_streams(
        self,
        min_ws_area: float = 1e6,
        min_stream_length: float = 0,
        memory: int | None = 4096,
    ):
        """Create the streams, flow direction and flow accumulation attributes.

        Args:
            min_ws_area (float, optional): Minimum contributing area (m2) used to
            classify streams. Defaults to 1e6 (1 km squared).
            min_stream_length (float, optional): Minimum length of stream segments.
            Defaults to 0.
            memory (Union[int, None], optional): Memory size to pass to GRASS for memory
            management of `r.watershed` operations. Defaults to 4096.
        """
        streams = self.raster_path("streams")
        fd = self.raster_path("flow_direction")
        fa = self.raster_path("flow_accumulation")

        flow_direction_accumulation(self.dem, fd, fa, memory=memory)

        extract_streams(
            self.dem,
            fa,
            streams,
            fd,
            min_ws_area,
            min_stream_length,
            memory=memory,
        )

        self.streams = streams
        self.flow_direction = fd
        self.flow_accumulation = fa
        self.min_ws_area = min_ws_area

    def calc_stream_attributes(self, focal_mean_dist: float | None = 200.0):
        """Calculate bankfull width, bankfull depth and channel gradient on stream
        cells.

        Args:
            focal_mean_dist (float | None, optional): Distance (m) over which channel
                gradient is averaged so it represents reaches rather than cells.
                Defaults to 200.
        """
        if self.streams is None or self.flow_direction is None:
            raise AttributeError(
                "Streams must be extracted using `RiparianConnectivity.extract_streams`"
                " prior to calculating stream attributes."
            )

        bankfull_width_dst = self.raster_path("bankfull_width")
        bankfull_depth_dst = self.raster_path("bankfull_depth")
        stream_slope_dst = self.raster_path("stream_slope")

        bankfull_width(
            self.streams, self.flow_direction, self.precip, bankfull_width_dst
        )
        bankfull_depth(bankfull_width_dst, bankfull_depth_dst)

        stream_slope(
            self.dem,
            self.streams,
            stream_slope_dst,
            units="percent",
            focal_mean_dist=focal_mean_dist,
        )

        self.bankfull_width = bankfull_width_dst
        self.bankfull_depth = bankfull_depth_dst
        self.stream_slope = stream_slope_dst

    def calc_zones(
        self,
        f1: float = 1.0,
        f2: float = 3.0,
        h3: float = 3.0,
        slope_breaks: tuple[float, float] = (0.015, 0.04),
        g_range: tuple[float, float] = (0.5, 1.25),
        t_min: float | None = None,
        cost_floor: float = DEFAULT_COST_FLOOR,
        fringe_mode: str = "rise",
        a_hill: float | None = None,
        gamma: float = 0.5,
        w_min: float = 0.25,
        memory: int | None = 4096,
    ):
        """Delineate the three riparian zones.

        Budgets (metres of rise) are assigned to each stream cell:

            g  = slope_modifier(channel gradient)
            T2 = max(f2 * D_bf * g, t_min * f2 / f1)
            T1 = T2 * f1 / f2
            T3 = T2 + h3

        and consumed along the least-climb path away from the stream. A cell belongs to
        the innermost zone whose budget exceeds the rise required to reach it.

        With `fringe_mode="convergence"` a second pass is seeded from the floodplain
        (zones 1 and 2) with friction reduced in convergent terrain
        (see `convergence_weight`), and the result is unioned with zone 3. This
        extends the fringe into hollows and toe slopes that receive hillslope
        groundwater.

        Args:
            f1 (float, optional): Bankfull depth multiple defining the channel and
                active floodplain (zone 1). Defaults to 1.0 (bankfull stage).
            f2 (float, optional): Bankfull depth multiple defining the flood-prone area
                (zone 2). Defaults to 3.0 (roughly Rosgen's flood-prone stage of twice
                the maximum bankfull depth).
            h3 (float, optional): Rise (m) above the zone 2 limit within which the
                water table is assumed to remain within rooting depth of the surface
                (zone 3). Equivalent to rooting depth / (1 - water table ratio); reduce
                in arid settings or where the water table is decoupled from topography.
                Defaults to 3.0.
            slope_breaks (tuple[float, float], optional): Channel gradients (m/m)
                bounding the gradient modifier. Defaults to (0.015, 0.04).
            g_range (tuple[float, float], optional): (min, max) of the gradient
                modifier. Defaults to (0.5, 1.25).
            t_min (float | None, optional): Minimum zone 1 budget (m). Defaults to
                twice the friction of a flat cell, which guarantees at least one cell
                beyond the channel.
            cost_floor (float, optional): Minimum friction as a gradient (m/m). Bounds
                the lateral extent on flat terrain to `budget / cost_floor`.
                Defaults to tan(0.5 degrees).
            fringe_mode (str, optional): `"rise"` derives zone 3 from the rise budget
                alone; `"convergence"` adds the convergence-weighted second pass.
                Defaults to "rise".
            a_hill (float | None, optional): Contributing area (m2) above which
                terrain is considered convergent. Defaults to a tenth of the stream
                initiation area used in `extract_streams`.
            gamma (float, optional): Rate at which fringe friction decreases with
                contributing area. Defaults to 0.5.
            w_min (float, optional): Minimum fringe friction weight. Defaults to 0.25.
            memory (int | None, optional): Memory (MB) for `r.cost`. Defaults to 4096.
        """
        if fringe_mode not in ("rise", "convergence"):
            raise ValueError('fringe_mode must be one of "rise" or "convergence".')
        if fringe_mode == "convergence" and self.flow_accumulation is None:
            raise AttributeError(
                "Flow accumulation is required for the convergence fringe; extract "
                "streams using `RiparianConnectivity.extract_streams`."
            )
        if self.streams is None:
            raise AttributeError(
                "Streams must be extracted using `RiparianConnectivity.extract_streams`"
                " prior to calculating zones."
            )
        if self.bankfull_depth is None or self.stream_slope is None:
            raise AttributeError(
                "Stream attributes must be calculated using "
                "`RiparianConnectivity.calc_stream_attributes` prior to calculating "
                "zones."
            )
        if not 0 < f1 < f2:
            raise ValueError("f1 and f2 must satisfy 0 < f1 < f2.")
        if h3 <= 0:
            raise ValueError("h3 must be positive.")

        if self.slope is None:
            slope_dst = self.raster_path("slope")
            slope(self.dem, slope_dst)
            self.slope = slope_dst

        avg_cs = (self.csx + self.csy) / 2.0
        if t_min is None:
            t_min = 2.0 * cost_floor * avg_cs

        stream_mask = ~da.ma.getmaskarray(from_raster(self.streams))
        d_bf = from_raster(self.bankfull_depth).astype(np.float64)

        # Channel gradient from percent to m/m; reaches without a value receive g = 1
        gradient = from_raster(self.stream_slope).astype(np.float64) / 100.0
        g = da.ma.filled(slope_modifier(gradient, slope_breaks, g_range), 1.0)

        # Streams without a depth receive the minimum budget so they remain seeds
        t2 = da.maximum(
            da.ma.filled(f2 * d_bf * g, 0.0), t_min * (float(f2) / float(f1))
        )
        t3 = da.ma.masked_where(~stream_mask, t2 + float(h3))

        t_max = float(t3.max().compute())

        start_cost_dst = self.raster_path("start_cost")
        to_raster(
            (t_max - t3).astype(np.float32),
            self.streams,
            start_cost_dst,
            as_cog=False,
        )

        cost_dst = self.raster_path("cost")
        nearest_dst = self.raster_path("nearest")
        rise_cost_surface(
            self.slope,
            start_cost_dst,
            cost_dst,
            nearest_dst,
            cost_floor=cost_floor,
            max_cost=t_max,
            memory=memory,
        )

        zones_dst = self.raster_path("zones")
        score_dst = self.raster_path("score")
        fringe_cost_dst = fringe_nearest_dst = None

        if fringe_mode == "rise":
            riparian_zones(
                cost_dst,
                nearest_dst,
                t_max,
                f1,
                f2,
                h3,
                zones_dst,
                score_dst,
                streams=self.streams,
            )
        else:
            if a_hill is None:
                if self.min_ws_area is None:
                    raise ValueError(
                        "a_hill must be provided when streams were not extracted with "
                        "`RiparianConnectivity.extract_streams`."
                    )
                a_hill = 0.1 * self.min_ws_area

            rise_zones_dst = self.raster_path("zones_rise")
            rise_score_dst = self.raster_path("score_rise")
            riparian_zones(
                cost_dst,
                nearest_dst,
                t_max,
                f1,
                f2,
                h3,
                rise_zones_dst,
                rise_score_dst,
                streams=self.streams,
            )

            # Seed the fringe pass from the floodplain, carrying the stream budgets
            rise_zones = from_raster(rise_zones_dst)
            floodplain = ~da.ma.getmaskarray(rise_zones) & (rise_zones <= 2)
            fringe_start_dst = self.raster_path("fringe_start_cost")
            to_raster(
                da.ma.masked_where(
                    ~floodplain, from_raster(nearest_dst).astype(np.float32)
                ),
                nearest_dst,
                fringe_start_dst,
                as_cog=False,
            )

            weight_dst = self.raster_path("fringe_weight")
            to_raster(
                convergence_weight(
                    from_raster(self.flow_accumulation),
                    self.csx * self.csy,
                    a_hill,
                    gamma,
                    w_min,
                ).astype(np.float32),
                self.flow_accumulation,
                weight_dst,
                as_cog=False,
            )

            fringe_cost_dst = self.raster_path("fringe_cost")
            fringe_nearest_dst = self.raster_path("fringe_nearest")
            rise_cost_surface(
                self.slope,
                fringe_start_dst,
                fringe_cost_dst,
                fringe_nearest_dst,
                cost_floor=cost_floor,
                max_cost=t_max,
                weight_src=weight_dst,
                memory=memory,
            )

            extend_fringe(
                fringe_cost_dst,
                fringe_nearest_dst,
                t_max,
                h3,
                rise_zones_dst,
                rise_score_dst,
                zones_dst,
                score_dst,
            )

        region_dst = self.raster_path("region")
        to_raster(
            ~da.ma.getmaskarray(from_raster(zones_dst)), zones_dst, region_dst
        )

        self.t_max = t_max
        self.start_cost = start_cost_dst
        self.cost = cost_dst
        self.nearest = nearest_dst
        self.fringe_cost = fringe_cost_dst
        self.fringe_nearest = fringe_nearest_dst
        self.zones = zones_dst
        self.score = score_dst
        self.region = region_dst

    def calc_connectivity(
        self,
        connectivity_dst: str,
        score_dst: str | None = None,
        vector: bool = True,
    ):
        """Save the riparian zones.

        Args:
            connectivity_dst (str): Output path for the zones. A polygon vector
                dataset with a zone attribute if `vector` is `True`, otherwise a raster
                with values 1 (channel and active floodplain), 2 (flood-prone area) and
                3 (groundwater-connected fringe).
            score_dst (str | None, optional): Output raster with the fraction of the
                rise budget remaining at each cell, which grades connectivity within
                zones. Defaults to None.
            vector (bool, optional): Save the zones as a vector file. Defaults to True.
        """
        if self.zones is None:
            raise AttributeError(
                "Zones must be calculated using `RiparianConnectivity.calc_zones` "
                "prior to saving connectivity."
            )

        if vector:
            vectorize(self.zones, connectivity_dst, smooth_corners=True)
        else:
            to_raster(from_raster(self.zones), self.zones, connectivity_dst)

        if score_dst is not None:
            to_raster(from_raster(self.score), self.score, score_dst)
