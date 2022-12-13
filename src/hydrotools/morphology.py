from typing import Union
from collections import OrderedDict
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool

import fiona
from shapely.geometry import LineString, Point
import dask.array as da
from dask_image.ndmorph import binary_dilation
from numba import njit, types
from numba.typed import List, Dict
import numpy as np

from hydrotools.raster import (
    Raster,
    TempRasterFiles,
    raster_where,
    from_raster,
    to_raster,
)
from hydrotools.utils import GrassRunner
from hydrotools.watershed import extract_streams, flow_direction_accumulation
from hydrotools.elevation import slope


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

    bankfull_width = bw_coeff * (contrib_area[km2] ** a_exp) * (precip[cm/yr] ** p_exp)

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
    da.ma.set_fill_value(elevation, -999)
    da.ma.set_fill_value(bf_elevation, -999)

    # Load stream locations and DEM into memory
    (_, i, j), complete, bfe, bfw, str_dist, elev = da.compute(
        da.where(~da.ma.getmaskarray(bf_elevation)),
        ~da.ma.getmaskarray(bf_elevation),
        da.ma.filled(bf_elevation),
        from_raster(bankfull_width_src),
        da.zeros_like(elevation, dtype="float32"),
        da.ma.filled(elevation),
    )

    raster_specs = Raster.raster_specs(dem)
    shape = raster_specs["shape"][1], raster_specs["shape"][2]
    csx, csy = raster_specs["csx"], raster_specs["csy"]

    stack = List(np.array([i, j]).T.tolist())
    del i, j

    @njit(fastmath=True, parallel=True)
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
    twi_src: str,
    valley_confinement_dst: str,
    twi_threshold: float = 3500,
):
    """Calculate a valley confinement index - the ratio of valley width to estimated
    bankfull width.

    Args:
        bankfull_width_src (str): Bankfull width data source calculated using
        `morphology.bankfull_width`.
        twi_src (str): Topographic Wetness Index data source calculated using
        `morphology.topographic_wetness`.
        valley_confinement_dst (str): Output raster with valley confinement index
        values.
        twi_threshold (float): The maximum topographic wetness used to constrain the
        valley bottom. Defaults to 3500.
    """

    @njit(fastmath=True)
    def calc_vci(
        stack,
        bfw,
        ve,
        vw,
    ):
        i_bound, j_bound = bfw.shape[1:]

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
            next_i, next_j = stack.pop()
            searched = {next_i: {next_j: True}}
            search_stack = List([next_i, next_j])

            while True:
                i, j = search_stack.pop(0)
                found = False

                for i_off, j_off in offsets:
                    i_nbr = i + i_off
                    j_nbr = j + j_off

                    if i_nbr == i_bound or i_nbr < 0 or j_nbr == j_bound or j_nbr < 0:
                        continue

                    try:
                        searched[i_nbr][j_nbr]
                        continue
                    except:
                        try:
                            searched[i_nbr][j_nbr] = True
                        except:
                            searched[i_nbr] = {j_nbr: True}
                        if ve[i_nbr, j_nbr]:
                            continue
                        if bfw[i_nbr, j_nbr]:
                            vw[i_nbr, j_nbr] += np.sqrt(
                                (next_i - i_nbr) ** 2 + (next_j - j_nbr) ** 2
                            )
                            found = True
                            break
                        else:
                            search_stack.append([i_nbr, j_nbr])

                if found:
                    break

        return np.where(vw > 0, bfw / vw, 0)

    valleys = from_raster(twi_src) >= twi_threshold
    valley_edges = valleys & binary_dilation(~valleys, np.ones((1, 3, 3), dtype=bool))

    bf_width = from_raster(bankfull_width_src)
    valley_width = da.zeros_like(bf_width)

    (_, i, j), bfw, ve, vw = da.compute(
        da.where(valley_edges), bf_width, valley_edges, valley_width
    )

    vci = da.from_array(calc_vci(List(np.array([i, j]).T.tolist()), bfw, ve, vw))
    vci = da.ma.masked_where(vci == 0, vci)

    to_raster(vci, bankfull_width_src, valley_confinement_dst)


def topographic_wetness(
    streams: str,
    slope_src: str,
    topographic_wetness_dst: str,
    cutoff: Union[float, None] = None,
):
    """Calculate a topographic wetness index

    Args:
        streams (str): Grid with stream locations
        (generated using `watershed.flow_direction_accumulation`)
        slope_src (str): Topographic slope in Degrees.
        topographic_wetness_dst (str): Output TWI grid
        cutoff (Union[float, None]): Return the index up to a defined threshold. Defaults to None.
    """
    streams = ~da.ma.getmaskarray(from_raster(streams))

    # Generate cost surface (normalized slope)
    cost = da.clip(from_raster(slope_src), 0.0, 90.0) / 90.0
    # Make cost 0 where simulated streams exist
    cost = raster_where(~streams, cost, 0)

    # Generate least cost surface
    with TempRasterFiles(3) as (streams_path, cost_path, cost_surface):
        to_raster(cost, slope_src, cost_path, False)
        to_raster(streams, slope_src, streams_path, False)

        with GrassRunner(cost_path) as gr:
            gr.run_command(
                "r.cost",
                (cost_path, "cost", "raster"),
                (streams_path, "streams", "raster"),
                input="cost",
                output="cost_path",
                start_raster="streams",
            )
            gr.save_raster("cost_path", cost_surface)

        cs = from_raster(cost_surface)

        # Invert
        cs_min = cs.min()
        cs_inverted = (cs.max() - cs_min) - (cs - cs_min)

        if cutoff is not None:
            cs_inverted = da.ma.masked_where(cs_inverted < float(cutoff), cs_inverted)

        cs_inverted = cs_inverted.astype("float32")

        to_raster(cs_inverted, slope_src, topographic_wetness_dst)


def riparian_connectivity(
    dem: str,
    annual_precip: str,
    connectivity_dst: str,
    min_ws_area: float = 1e6,
    **kwargs
):
    """Calculate regions of riparian connectivity surrounding streams

    Args:
        dem (str): Input Digital Elevation Model raster.
        annual_precip (str): Grid of annual precipitation in mm.
        connectivity_dst (str): Output vector dataset with categorized riparian
        connectivity zones ranging from 1 to 3.
        min_ws_area: Minimum contributing area (m2) used to classify streams. Defaults
        to 1e6 (1 km squared).
    """
    dem_rast = Raster(dem)
    if not np.isclose(dem_rast.csx - dem_rast.csy, 0):
        raise ValueError("Input grid must be isotropic")

    with TempRasterFiles(8) as (
        accumulation_dst,
        direction_dst1,
        direction_dst2,
        stream_dst,
        slope_dst,
        bankfull_dst,
        cost_dst,
        lcp_dst,
    ):
        # Derive streams
        flow_direction_accumulation(
            dem,
            direction_dst1,
            accumulation_dst,
            False,
            False,
        )

        extract_streams(
            dem,
            accumulation_dst,
            stream_dst,
            direction_dst2,
            min_ws_area,
            kwargs.get("min_length", 0),
        )

        # Create a cost surface
        # Source regions are comprised of Bankfull Width
        bankfull_width(
            accumulation_dst,
            annual_precip,
            bankfull_dst,
            streams=stream_dst,
            distribute=True,
        )

        slope(dem, slope_dst, overviews=False)

        cost = raster_where(
            da.ma.getmaskarray(from_raster(bankfull_dst)),
            da.clip(from_raster(slope_dst), 0.0, 90.0) / 90.0,
            0,
        )

        to_raster(cost, slope_dst, cost_dst, False)

        with GrassRunner(cost_dst) as gr:
            gr.run_command(
                "r.cost",
                (cost_dst, "cost", "raster"),
                (stream_dst, "streams", "raster"),
                input="cost",
                output="conn",
                start_raster="streams",
            )
            # gr.save_raster("conn", lcp_dst)
            gr.save_raster("conn", connectivity_dst)

        # Classify connectivity region
        # conn_threshold = kwargs.get("lcp_threshold", 1)
        # conn_region = from_raster(lcp_dst) < conn_threshold

        # Calculate other variables and interpolate to the connectivity region
        # Channel density
        # convolve(stream_dst, np.ones((10, 10, 1), dtype="bool"), "sum")

        # Inverse of normalized channel slope
        # to_raster(raster_where(da.ma.getmaskarray(from_raster(stream_dst)), from_raster(dem), None))
        # slope(strem_elev_dst, stream_slope_dst, overviews=False)
        # 1 - da.clip(from_raster(stream_slope_dst), 0.0, 90.0) / 90

        # precip_cm = from_raster(precip).astype("float32") / 10

        # bankfull_width = 0.196 * (contrib_area**0.280) * (precip_cm**0.355)

        # Cost (already distributed)
        # 1 - da.clip(raster_where(from_raster(connectivity_dst), 0, conn_threshold) / conn_threshold

        # Classify into 3 zones based on distribution
