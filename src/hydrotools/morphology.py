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

from hydrotools.config import CHUNKS, TMP_DIR
from hydrotools.elevation import slope
from hydrotools.interpolate import distance_transform, raster_filter
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

            bfw = np.full(stream_mask.shape[1:], -1, np.float32)
            bfw[(stream_inds[indices][:, 0], stream_inds[indices][:, 1])] = distances
            avg_cs = (raster_specs["csx"] + raster_specs["csy"]) / 2
            bfw[bfw == 0] = avg_cs

            # Interpolate where streams are missing values
            valid_bfw = bfw != -1
            points = np.array(np.where(valid_bfw)).T
            values = bfw[valid_bfw]
            xi = np.array(np.where(~valid_bfw & stream_mask.compute()[0])).T

            bfw[(xi[:, 0], xi[:, 1])] = griddata(points, values, xi, fill_value=avg_cs)

            out_dask = da.from_array(bfw.reshape(stream_mask.shape)).rechunk(
                stream_mask.chunks
            )

            to_raster(
                da.ma.masked_where(out_dask == -1, out_dask),
                lcp_path,
                bankfull_dst,
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




class RiparianConnectivity:
    """Riparian analysis workflow

    Riparian connectivity is modelled as a least-cost expansion from streams. Each
    stream cell is assigned a riparian reach (m) derived from its bankfull width and
    stream slope, which is used as the initial cost in `r.cost` (using the `-r` flag).
    The reach is consumed by a terrain cost surface expressed in metres of effective
    distance (horizontal distance plus a weighted vertical climb), so the riparian
    region is a continuation of the magnitudes defined by bankfull width.

    The general steps are:

    1. Instantiate using DEM

        `rc = RiparianConnectivity(dem_path)`

    2. Extract streams

        `rc.extract_streams()`

    3. Calculate the stream attributes used to define the riparian reach:

        ```
        rc.calc_bankfull_width(annual_precip_grid)
        rc.calc_stream_slope()
        ```

    4. Define a riparian region

        `rc.define_region()`

        Here you should visually examine the output region (the path to which can be
        accessed using the `region` attribute) and verify it aligns with valley
        bottoms. If needed, recalculate using different parameters. Ex.
        `rc.define_region(width_factor=4, vertical_weight=30)`

    5. Classify the region into Riparian Connectivity zones:

        `rc.calc_connectivity(output_dataset)`

    """

    def __init__(self, dem: str):
        """Define a study area and begin a Riparian Connectivity Analysis"""
        self.dem = dem

        # Specifications
        self.__dict__.update(Raster.raster_specs(dem))

        # Create a directory
        self.tmp_dir = TemporaryDirectory(dir=TMP_DIR)

        # Components of the analysis
        (
            self.slope,
            self.flow_direction,
            self.streams,
            self.bankfull_width,
            self.stream_slope,
            self.region,
            self.reach_budget,
        ) = (None for _ in range(7))

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

    # TODO: copy tmp_dir to a specified dir
    # def save(self, dst_dir: str):
    #     """_summary_

    #     Args:
    #         dst_dir (str): _description_
    #     """

    # @classmethod
    # def load_from_data(cls):
    # TODO: Load from all required parameters

    def extract_streams(
        self,
        min_ws_area: float = 1e6,
        min_stream_length: float = 0,
        memory: int | None = 4096,
    ):
        """Create a streams attribute.

        Args:
            min_ws_area (float, optional): Minimum contributing area (m2) used to
            classify streams. Defaults to 1e6 (1 km squared).
            min_stream_lenth (float, optional): Minimum length of stream segments.
            Defaults to 0.
            memory (Union[int, None], optional): Memory size to pass to GRASS for memory
            management of `r.watershed` operations. Defaults to None.
        """
        streams = self.raster_path("streams")
        if self.flow_direction is None:
            fd2 = self.raster_path("flow_direction")
        else:
            fd2 = self.flow_direction

        with TempRasterFiles(2) as (fd, fa):
            if self.flow_direction is None:
                flow_direction_accumulation(self.dem, fd, fa, memory=memory)

            extract_streams(
                self.dem,
                fa,
                streams,
                fd2,
                min_ws_area,
                min_stream_length,
                memory=memory,
            )

        self.streams = streams
        self.flow_direction = fd2

    def calc_slope(self):
        """Calculate a topographic slope (degrees) attribute, if it does not exist."""
        if self.slope is None:
            slope_dst = self.raster_path("slope")
            slope(self.dem, slope_dst)
            self.slope = slope_dst

    def calc_bankfull_width(self, annual_precip_src: str):
        """Calculate a Bankfull Width (m) attribute along streams.

        Args:
            annual_precip_src (str): Mean annual precipitation grid in mm.
        """
        if self.streams is None or self.flow_direction is None:
            raise AttributeError(
                "Streams must be extracted using `RiparianConnectivity.extract_streams`"
                " prior to calculating Bankfull Width."
            )

        self.verify_source(annual_precip_src)

        bankfull_width_dst = self.raster_path("bankfull_width")
        bankfull_width(
            self.streams, self.flow_direction, annual_precip_src, bankfull_width_dst
        )

        self.bankfull_width = bankfull_width_dst

    def calc_stream_slope(self, focal_mean_dist: None | float = None):
        """Calculate a Stream Slope (degrees) attribute along streams.

        Args:
            focal_mean_dist (Union[None, float]): Sampling distance used to smooth
                stream slope values. See `morphology.stream_slope`. Defaults to None.
        """
        if self.streams is None:
            raise AttributeError(
                "Streams must be extracted using `RiparianConnectivity.extract_streams`"
                " prior to calculating Stream Slope."
            )

        stream_slope_dst = self.raster_path("stream_slope")
        stream_slope(
            self.dem, self.streams, stream_slope_dst, focal_mean_dist=focal_mean_dist
        )

        self.stream_slope = stream_slope_dst

    def define_region(
        self,
        width_factor: float = 3.0,
        slope_cutoff: float = 15.0,
        vertical_weight: float = 20.0,
        max_reach: float = 100.0,
        knights_move: bool = True,
        memory: int | None = 4096,
    ):
        """Delineate the riparian region and its connectivity using a least-cost
        expansion from streams.

        Each stream cell is assigned a riparian reach (m):

            reach = width_factor * bankfull_width * (1 - min(stream_slope, slope_cutoff)
                / slope_cutoff)

        The reach is capped at `max_reach`, converted to an initial cost of
        `max_reach - reach`, and passed to `r.cost` using the `-r` flag. The terrain
        cost of traversing a cell is an effective distance (m):

            cost = cell_size * (1 + vertical_weight * tan(slope))

        which is the horizontal distance travelled plus the vertical distance climbed,
        weighted by `vertical_weight`. Cells with an accumulated cost less than
        `max_reach` are within the riparian region, and the remaining reach, normalized
        by `max_reach`, is saved to the `region` attribute (1 is the most connected).
        The remaining reach in metres may be calculated using `region * reach_budget`.

        Because `max_reach` is a fixed value rather than derived from the study area,
        `region` values are comparable between watersheds analyzed with the same
        parameters.

        This method may be called iteratively, inspecting the result to determine
        whether the extent accurately represents the extent of riparian regions.

        Args:
            width_factor (float, optional): Multiple of bankfull width used to define
                the riparian reach on each side of a stream. Defaults to 3.0.
            slope_cutoff (float, optional): Stream slope (degrees) at which the
                riparian reach is reduced to zero. Defaults to 15.0.
            vertical_weight (float, optional): Effective horizontal distance (m)
                incurred per metre of vertical climb. Defaults to 20.0.
            max_reach (float, optional): Maximum riparian reach (m), used as a fixed
                budget to normalize the region. Keep this constant between watersheds
                that will be compared. Defaults to 100.0.
            knights_move (bool, optional): Use the knight's move in `r.cost`.
                Defaults to True.
            memory (Union[int, None], optional): Maximum memory usage for GRASS
                operations. Defaults to 4096.
        """
        if self.bankfull_width is None or self.stream_slope is None:
            raise AttributeError(
                "Bankfull Width and Stream Slope must be calculated using "
                "`RiparianConnectivity.calc_bankfull_width` and "
                "`RiparianConnectivity.calc_stream_slope` prior to defining the "
                "riparian region."
            )
        if slope_cutoff <= 0:
            raise ValueError("The slope cutoff must be greater than 0.")
        if max_reach <= 0:
            raise ValueError("The maximum reach must be greater than 0.")
        budget = float(max_reach)

        self.calc_slope()

        avg_cs = (self.csx + self.csy) / 2.0

        stream_mask = ~da.ma.getmaskarray(from_raster(self.streams))
        terrain_slope = from_raster(self.slope).astype(np.float64)
        terrain_mask = da.ma.getmaskarray(terrain_slope)
        terrain_slope = da.ma.filled(terrain_slope, 0)

        # Stream slope gaps fall back to the terrain slope
        strslo = from_raster(self.stream_slope).astype(np.float64)
        strslo = da.where(
            da.ma.getmaskarray(strslo), terrain_slope, da.ma.filled(strslo, 0)
        )

        bfw = da.ma.filled(from_raster(self.bankfull_width).astype(np.float64), 0)

        # Riparian reach at streams
        reach = (
            width_factor
            * da.clip(bfw, 0, None)
            * (1.0 - da.clip(strslo, 0, slope_cutoff) / slope_cutoff)
        )
        reach = da.where(stream_mask, da.clip(reach, 0, budget), 0)

        start = da.ma.masked_where(~stream_mask, budget - reach)

        # Effective distance (m) cost surface
        cost = (
            1.0
            + vertical_weight * da.tan(da.deg2rad(da.clip(terrain_slope, 0, 89.0)))
        ) * avg_cs
        cost = da.ma.masked_where(terrain_mask, cost)

        region_dst = self.raster_path("region")
        with TempRasterFiles(3) as (start_path, cost_path, cost_surface):
            to_raster(start, self.slope, start_path, as_cog=False)
            to_raster(cost, self.slope, cost_path, as_cog=False)

            kwargs = {
                "input": "cost",
                "output": "cost_path",
                "start_raster": "start",
                "max_cost": budget,
                "flags": "kr" if knights_move else "r",
            }

            if memory is not None:
                kwargs["memory"] = memory

            with GrassRunner(cost_path) as gr:
                gr.run_command(
                    "r.cost",
                    (cost_path, "cost", "raster"),
                    (start_path, "start", "raster"),
                    **kwargs,
                )
                gr.save_raster("cost_path", cost_surface)

            cs = from_raster(cost_surface).astype(np.float32)
            region = (budget - da.ma.masked_where(cs > budget, cs)) / budget

            to_raster(region, self.slope, region_dst)

        self.region = region_dst
        self.reach_budget = budget

    def calc_connectivity(
        self,
        connectivity_dst: str,
        quantize: bool = True,
        breaks: list = [1.0 / 3.0, 2.0 / 3.0],
        percentiles: None | list = None,
        vector: bool = True,
    ):
        """Calculate a riparian connectivity dataset from the riparian region.

        Args:
            connectivity_dst (str): Output path for Riparian Connectivity. This will be
                a raster if `vector` is `False`.
            quantize (bool, optional): Split the result into zones of connectivity.
                Defaults to True.
            breaks (list, optional): Fixed region values (0-1) used to define zones of
                connectivity, which are comparable between watersheds. Defaults to
                [1.0 / 3.0, 2.0 / 3.0].
            percentiles (Union[None, list], optional): Percentiles used to define zones
                of connectivity in place of `breaks`. Zones are relative to the study
                area and are not comparable between watersheds. Defaults to None.
            vector (bool, optional): Save the output as a vector file.
        """
        if self.region is None:
            raise AttributeError(
                "The riparian region must be delineated using "
                "`RiparianConnectivity.define_region` prior to calculating "
                "connectivity."
            )

        thresholds = breaks if percentiles is None else percentiles
        upper = 1 if percentiles is None else 100
        if quantize and (
            len(thresholds) == 0
            or any([q2 < q1 for q1, q2 in zip(thresholds[0:-1], thresholds[1:])])
            or any([q < 0 or q > upper for q in thresholds])
        ):
            raise ValueError(
                "Breaks or percentiles must have at least one value, be monotonically "
                f"increasing, and be within 0 and {upper}."
            )

        if vector and not quantize:
            raise ValueError("A vector should only be specified if `quantize` is used.")

        connectivity = from_raster(self.region)

        if quantize:
            if percentiles is None:
                qs = breaks
            else:
                sample = connectivity[~da.ma.getmaskarray(connectivity)]
                qs = [da.percentile(sample, [q])[0] for q in percentiles]

            connectivity = da.ma.masked_where(
                da.ma.getmaskarray(connectivity),
                da.digitize(connectivity, qs).astype(np.uint8),
            )

        if vector:
            with TempRasterFile() as tmp_rast:
                to_raster(connectivity, self.dem, tmp_rast, as_cog=False)
                vectorize(tmp_rast, connectivity_dst, smooth_corners=True)
        else:
            to_raster(connectivity, self.dem, connectivity_dst)
