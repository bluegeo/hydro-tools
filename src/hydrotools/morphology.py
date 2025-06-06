import os
from typing import Union
from collections import OrderedDict
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from tempfile import TemporaryDirectory

import fiona
from shapely.geometry import LineString, Point
import dask.array as da
from dask_image.ndmorph import binary_erosion
from numba import njit
from numba.typed import List
from scipy.interpolate import griddata
import numpy as np
from sklearn.neighbors import BallTree

from hydrotools.config import CHUNKS
from hydrotools.raster import (
    Raster,
    raster_where,
    from_raster,
    to_raster,
    vectorize,
)
from hydrotools.utils import (
    GrassRunner,
    TempRasterFile,
    TempRasterFiles,
    kernel_from_distance,
    compare_projections,
)
from hydrotools.watershed import extract_streams, flow_direction_accumulation
from hydrotools.elevation import slope
from hydrotools.interpolate import (
    raster_filter,
    PointInterpolator,
    raster_filter,
    distance_transform,
    normalize,
)


def sinuosity(
    stream_vector: str, sinuosity_vector: str, sampling: Union[float, str] = 100
):
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
                out_vect.write(
                    {
                        "geometry": {
                            "type": "Point",
                            "coordinates": (point[0], point[1]),
                        },
                        "properties": OrderedDict([("sinuosity", point[2])]),
                    }
                )


def stream_slope(
    dem: str,
    streams: str,
    slope_dst: str,
    units: str = "degrees",
    scale: float = 1,
    focal_mean_dist: Union[None, float] = None,
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

                # Assign widths to output streams
                locations = np.ravel_multi_index(
                    (stream_inds[indices][:, 0], stream_inds[indices][:, 1]),
                    stream_mask.shape[1:],
                )

                bfw = da.zeros(
                    stream_mask.shape, dtype="float32", chunks=stream_mask.chunks
                ).ravel()
                bfw[locations] = distances
                bfw = bfw.reshape(stream_mask.shape).rechunk(stream_mask.chunks)
                bfw = da.ma.masked_equal(bfw, 0)

                # Interpolate where streams are missing values
                points = np.array(da.where(bfw[0])).T
                values = bfw[~da.ma.getmaskarray(bfw)].compute()
                xi = np.array(da.where((da.ma.getmaskarray(bfw) & stream_mask)[0])).T

                locations = np.ravel_multi_index(
                    (xi[:, 0], xi[:, 1]),
                    stream_mask.shape[1:],
                )
                bfw = bfw.ravel()
                avg_cs = (raster_specs["csx"] + raster_specs["csy"]) / 2
                bfw[locations] = griddata(points, values, xi, fill_value=avg_cs)

                to_raster(
                    bfw.reshape(stream_mask.shape).rechunk(stream_mask.chunks),
                    lcp_path,
                    bankfull_dst,
                )


def valley_width(
    streams: str,
    slope: str,
    dst: str,
    lcp_dst: str = None,
    max_cost: float = 1,
    memory: int = None,
):
    """Mirror of bankfull_width_geometric"""
    return bankfull_width_geometric(streams, slope, dst, lcp_dst, max_cost, memory)


def bankfull_width(
    streams: str,
    accumulation: str,
    precip: str,
    bankfull: str,
    bw_coeff: float = 0.196,
    a_exp: float = 0.280,
    p_exp: float = 0.355,
):
    """Estimate bankfull width with empirically-derived constants using:

    bankfull_width = bw_coeff * \
        (contrib_area[km2] ** a_exp) * (precip[cm/yr] ** p_exp)

    Method and defaults from:

    Hall, J. E., D. M. Holzer, and T. J. Beechie. 2007. Predicting river floodplain and
        lateral channel migration for salmon habitat conservation. Journal of the
        American Water Resources Association 43:786-797.

    Note, ensure the spatial reference of the rasters is a projected system in metres.

    An example workflow that starts with a DEM and uses streams with a minumum
    contributing area could look like this:

    ```python
    with TempRasterFile() as tmp_flow_direction:
        # Compute flow accumulation. The resulting flow direction is not needed here
        flow_direction_accumulation(
            dem_path,
            tmp_flow_direction,
            accumulation_dst,
            False,  # Multiple flow-direction
            False,  # Allow negatives
        )

        # Compute streams with a minimum length of 0 while saving flow direction
        extract_streams(
            dem_path, accumulation_dst, stream_dst, flow_direction_dst, 1e6, 0
        )

        bankfull_width(
            stream_dst,
            accumulation_dst,
            avg_annual_precip,
            bankfull_dst,
        )
    ```

    Args:
        streams: Grid with stream locations
        (generated using `watershed.extract_streams`),
        accumulation (str): Flow accumulation dataset
        (generated using `watershed.flow_direction_accumulation`)
        precip (str): Mean annual precipitation grid in mm
        bankfull (str): Destination raster for bankfull grid
        streams (Union[str, None], optional): Input streams raster. Defaults to None.
        bw_coeff (float): Bankfull width coefficient. Defaults to 0.196.
        a_exp (float): Area expenential scale. Defaults to 0.280.
        p_exp (float): Precip expenential scale. Defaults to 0.355.
    """
    # Contributing area in km**2
    ca_specs = Raster(accumulation)
    fa = from_raster(accumulation)
    contrib_area = da.ma.masked_where(
        da.ma.getmaskarray(from_raster(streams)),
        da.abs(fa.astype("float32")) * ((ca_specs.csx * ca_specs.csy) / 1e6),
    )

    # Convert precipitation to cm
    precip_cm = from_raster(precip).astype("float32") / 10

    bankfull_width = bw_coeff * (contrib_area**a_exp) * (precip_cm**p_exp)

    to_raster(
        da.ma.masked_where(
            da.isinf(bankfull_width) | da.isnan(bankfull_width), bankfull_width
        ),
        accumulation,
        bankfull,
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

    The general steps are:

    1. Instantiate using DEM

        `rc = RiparianConnectivity(dem_path)`

    2. Extract streams

        `rc.extract_streams()`

    3. Calculate the Topographic Wetness Index

        `rc.calc_twi()`

    4. Define a riparian region

        `rc.define_region()`

        Here you should visually examing the output regions and verify they align with
        valley bottoms. If needed, recalculate using a different twi_cutoff. Ex.
        `rc.define_region(740)`

    5. You may now calculate attributes that you wish to add. The following
        includes all of them:

        ```
        rc.calc_bankfull_width(annual_precip_grid)
        rc.calc_stream_slope()
        rc.calc_stream_density()
        ```

    6. Calculate Riparian Connectivity regions using the desired attributes:

        `rc.calc_connectivity(output_dataset)`

    """

    def __init__(self, dem: str):
        """Define a study area and begin a Riparian Connectivity Analysis"""
        self.dem = dem

        # Specifications
        self.__dict__.update(Raster.raster_specs(dem))

        # Create a directory
        self.tmp_dir = TemporaryDirectory()

        # Components of the analysis
        (
            self.slope,
            self.flow_accumulation,
            self.streams,
            self.twi,
            self.region,
            self.bankfull_width,
            self.stream_slope,
            self.stream_density,
        ) = (None for _ in range(8))

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
        if not all(
            [
                np.isclose(src_specs.csx, self.csx),
                np.isclose(src_specs.csy, self.csy),
                np.isclose(src_specs.top, self.top),
                np.isclose(src_specs.bottom, self.bottom),
                np.isclose(src_specs.left, self.left),
                np.isclose(src_specs.right, self.right),
                src_specs.shape == self.shape,
                compare_projections(src_specs.wkt, self.wkt),
            ]
        ):
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
        memory: Union[int, None] = 4096,
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
        if self.flow_accumulation is None:
            fa = self.raster_path("flow_accumulation")
        else:
            fa = self.flow_accumulation

        with TempRasterFiles(2) as (fd1, fd2):
            if self.flow_accumulation is None:
                flow_direction_accumulation(
                    self.dem, fd1, fa, False, False, memory=memory
                )

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
        self.flow_accumulation = fa

    def calc_twi(self, memory: int = 4096):
        """Calculate a Topographic Wetness Attribute, which is also used to constrain the
        riparian extent.
        """
        if self.streams is None:
            raise AttributeError(
                "Streams must be delineated first using "
                "`RiparianConnectivity.extract_streams`."
            )
        if self.slope is None:
            slope_dst = self.raster_path("slope")
            slope(self.dem, slope_dst)
            self.slope = slope_dst

        twi_dst = self.raster_path("twi")
        topographic_wetness(self.streams, self.slope, twi_dst, memory=memory)
        self.twi = twi_dst

    def define_region(self, twi_cutoff: float = 745.0):
        """Constrain the Topographic Wetness Index (TWI) to a threshold that is used
        to define the extent of the riparian.

        This method may be called iteratively after calculating the TWI, defining the
        region using a threshold, and inspecting the result (the path to which can be
        accessed using the `region` attribute) to determine whether the extent
        accurately represents the extent of riparian regions.

        Args:
            twi_cutoff (float): Topographic wetness cutoff value. Defaults to 830.
        """
        if self.twi is None:
            raise AttributeError(
                "The Topographic Wetness Index must be calculated "
                "first using `RiparianConnectivity.calc_twi`."
            )

        twi = from_raster(self.twi)
        twi = da.ma.masked_where(twi < float(twi_cutoff), twi)

        twi_max = twi.max()
        twi_norm = (twi - twi_cutoff) / (twi_max - twi_cutoff)

        region_dst = self.raster_path("region")
        to_raster(twi_norm, self.twi, region_dst)

        self.region = region_dst

    def interpolate_into_region(self, src, dst):
        obs_raster = Raster(src)
        obs, obs_points = Raster(src).data_and_index()

        obs_points = np.asarray(obs_points, dtype="float32")

        region_values, pred_points = Raster(self.region).data_and_index()

        del region_values

        pred_points = np.asarray(pred_points, dtype="float32")

        pred_z = PointInterpolator(
            obs_points,
            obs,
            pred_points,
        ).idw(int(round(200 / ((self.csx + self.csy) / 2.0))))

        nodata = np.finfo("float32").min
        output = np.full(obs_raster.shape[1:], nodata, "float32")

        pred_points = pred_points.astype("int64")
        output[(pred_points[:, 0], pred_points[:, 1])] = pred_z

        del pred_points, pred_z

        obs_points = obs_points.astype("int64")

        output[(obs_points[:, 0], obs_points[:, 1])] = obs

        del obs, obs_points

        output_dask = da.from_array(
            output.reshape(obs_raster.shape)
        ).rechunk(CHUNKS)

        to_raster(da.ma.masked_where(output_dask == nodata, output_dask), src, dst)

    def calc_bankfull_width(self, annual_precip_src: str, cutoff: float = 10.0):
        """Calculate a Bankfull Width attribute, which is normalized and  expanded to
        the riparian region.

        Args:
            annual_precip_src (str): Mean annual precipitation grid in mm.
            cutoff (float, optional): A maximum Bankfull Width threshold to use for the
                normalized value. Defaults to 10.0m.
        """
        if self.streams is None or self.flow_accumulation is None:
            raise AttributeError(
                "Streams must be extracted using `RiparianConnectivity.extract_streams`"
                " prior to calculating Bankfull Width."
            )
        if self.region is None:
            raise AttributeError(
                "Riparian regions must be delineated using "
                "`RiparianConnectivity.define_region` prior to calculating Bankfull "
                "Width."
            )

        self.verify_source(annual_precip_src)

        bankfull_width_dst = self.raster_path("bankfull_width")
        with TempRasterFiles(2) as (bankfull_dst, bankfull_norm_dst):
            bankfull_width(
                self.streams, self.flow_accumulation, annual_precip_src, bankfull_dst
            )
            normalize(bankfull_dst, bankfull_norm_dst, (0, cutoff))
            self.interpolate_into_region(bankfull_norm_dst, bankfull_width_dst)

        self.bankfull_width = bankfull_width_dst

    def calc_stream_slope(self, cutoff: float = 15.0):
        """Calculate and add a Stream Slope attribute.

        Args:
            cutoff (float, optional): A maximum Stream Slope threshold to use for the
                normalized value. Defaults to 15.0 degrees.
        """
        if self.streams is None:
            raise AttributeError(
                "Streams must be extracted using `RiparianConnectivity.extract_streams`"
                " prior to calculating Stream Slope."
            )
        if self.region is None:
            raise AttributeError(
                "Riparian regions must be delineated using "
                "`RiparianConnectivity.define_region` prior to calculating Stream "
                "Slope."
            )

        stream_slope_dst = self.raster_path("stream_slope")

        with TempRasterFiles(2) as (strslo_dst, strslo_norm_dst):
            stream_slope(self.dem, self.streams, strslo_dst)
            normalize(
                strslo_dst,
                strslo_norm_dst,
                (0, cutoff),
                invert=True,
            )
            self.interpolate_into_region(strslo_norm_dst, stream_slope_dst)

        self.stream_slope = stream_slope_dst

    def calc_stream_density(self, sample_distance: float = 500, cutoff: float = 2e5):
        """Calculate and add a Stream Density attribute.

        Args:
            sample_distance (float, optional): Distance to sample stream cells for the
                stream density transform. Defaults to 500m.
            cutoff (float, optional): A maximum Stream Slope threshold to use for the
                normalized value. This value should assume a cell size of 1m, be
                adjusted with the `sample_distance` argument, and be adjusted for
                the actual input raster cell size. Defaults to 2e5 cells per the
                `radius` parameter.
        """
        if self.streams is None:
            raise AttributeError(
                "Streams must be extracted using `RiparianConnectivity.extract_streams`"
                " prior to calculating Stream Density."
            )
        if self.region is None:
            raise AttributeError(
                "Riparian regions must be delineated using "
                "`RiparianConnectivity.define_region` prior to calculating Stream "
                "Density."
            )
        # Would check for slope, but it will be implicitly included if
        # RiparianConnectivity.define_region has been called.

        stream_density_dst = self.raster_path("stream_density")
        with TempRasterFiles(4) as (
            stream_mask_dst,
            stream_filter_dst,
            strdens_dst,
            strdens_norm_dst,
        ):
            to_raster(
                ~da.ma.getmaskarray(from_raster(self.streams)),
                self.streams,
                stream_mask_dst,
                as_cog=False,
            )
            kernel = kernel_from_distance(sample_distance, self.csx, self.csy)
            raster_filter(stream_mask_dst, kernel, "sum", stream_filter_dst)

            stream_density = from_raster(stream_filter_dst) / kernel.sum()

            to_raster(
                stream_density.astype(np.float32),
                stream_filter_dst,
                strdens_dst,
                as_cog=False,
            )

            # Adjust stream_density_cutoff for cell size
            avg_cs = (self.csx + self.csy) / 2.0
            cutoff /= avg_cs**2
            normalize(strdens_dst, strdens_norm_dst, (0, cutoff))

            self.interpolate_into_region(strdens_norm_dst, stream_density_dst)

        self.stream_density = stream_density_dst

    def calc_connectivity(
        self,
        connectivity_dst: str,
        attributes: list = ["twi", "bankfull_width", "stream_slope", "stream_density"],
        quantize: bool = True,
        percentiles: list = [(1.0 / 3.0) * 100.0, (2.0 / 3.0) * 100.0],
        vector: bool = True,
    ):
        """Calculate a riparian connectivity dataset.

        Args:
            connectivity_dst (str): Output path for Riparian Connectivity. This will be
                a raster if `vector` is `False`.
            attributes (list, optional): Attributes to include in the riparian
                connectivity calculation. Defaults to
                ["bankfull_width", "stream_slope", "stream_density"].
            quantize (bool, optional): Split the result into zones of sensitivity using.
                percentiles. Defaults to True.
            percentiles (list, optional): Percentiles to use to define zones of
                connectivity. Defaults to [(1.0 / 3.0) * 100.0, (2.0 / 3.0) * 100.0].
            vector (bool, optional): Save the output as a vector file.
        """
        if len(attributes) == 0:
            raise ValueError("At least one attribute must be included")

        if quantize and (
            len(percentiles) == 0
            or any([q2 < q1 for q1, q2 in zip(percentiles[0:-1], percentiles[1:])])
            or any([q > 100 for q in percentiles])
        ):
            raise ValueError(
                "Percentiles must have at least one value, be monotonically increasing, "
                "and not exceed 100."
            )

        if vector and not quantize:
            raise ValueError("A vector should only be specified if `quantize` is used.")

        prepared_attrs = []
        for attr in attributes:
            if attr == "twi":
                attr = "region"

            attr_value = getattr(self, attr)
            if attr_value is None:
                raise AttributeError(f"The attribute '{attr}' has not been calculated.")
            prepared_attrs.append(attr_value)

        sensitivity = from_raster(prepared_attrs[0])
        for attr in prepared_attrs[1:]:
            sensitivity += from_raster(attr)

        if quantize:
            percentile_sample = sensitivity[~da.ma.getmaskarray(sensitivity)]
            qs = [da.percentile(percentile_sample, q)[0] for q in percentiles]

            sensitivity = da.ma.masked_where(
                da.ma.getmaskarray(from_raster(self.region)),
                da.digitize(sensitivity, qs).astype(np.uint8),
            )

        if vector:
            with TempRasterFile() as tmp_rast:
                to_raster(sensitivity, self.dem, tmp_rast, as_cog=False)
                vectorize(tmp_rast, connectivity_dst, smooth_corners=True)
        else:
            to_raster(sensitivity, self.dem, connectivity_dst)
