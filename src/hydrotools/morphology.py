from typing import Union
from collections import OrderedDict
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool

import fiona
from shapely.geometry import LineString, Point
import dask.array as da
from dask_image.ndmorph import binary_erosion
from numba import njit
from numba.typed import List
import numpy as np

from hydrotools.raster import (
    Raster,
    TempRasterFile,
    TempRasterFiles,
    raster_where,
    from_raster,
    to_raster,
)
from hydrotools.utils import GrassRunner
from hydrotools.watershed import extract_streams, flow_direction_accumulation
from hydrotools.elevation import slope
from hydrotools.interpolate import front_contains


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
    dem: str, streams: str, slope_dst: str, units: str = "degress", scale: float = 1
):
    """Calculate slope along extracted streams.

    Args:
        dem (str): Digital Elevation model raster
        streams (str): Streams raster source generated using `watershed.extract_streams`
        slope_dst (str): Destination slope raster dataset.
        units (str, optional): Units for the output. Defaults to "degrees".
        scale (float, optional): Z-factor to scale the output. Defaults to 1.
    """
    with TempRasterFile() as elev_dst:
        to_raster(
            da.ma.masked_where(
                da.ma.getmaskarray(from_raster(streams)), from_raster(dem)
            ),
            dem,
            elev_dst,
            overviews=False,
        )

        slope(elev_dst, slope_dst, units, scale, overviews=False)


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


@njit
def vci_width_transform_task(
    regions: np.ndarray, edges: np.ndarray, bfw: np.ndarray, csx: float, csy: float
) -> np.ndarray:
    """Compute approximate valley confinement index using widths of connected regions
    and returning the ratio of the maximum bankfull width to the valley width.

    Args:
        regions (np.ndarray): Array of regions (float) bounded by a value of -999.
        edges (np.ndarray): Array of edges (bool) of regions.
        bfw (np.ndarray): Array of bankfull width values, with -999 outside of streams.
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

        bfw_value = 0
        for assign_i, j_dict in searched.items():
            for assign_j, _ in j_dict.items():
                if (
                    bfw[assign_i, assign_j] != -999
                    and bfw[assign_i, assign_j] > bfw_value
                ):
                    bfw_value = bfw[assign_i, assign_j]

        if current_distance > 0:
            vci = bfw_value / current_distance
            for assign_i, j_dict in searched.items():
                for assign_j, _ in j_dict.items():
                    width[assign_i, assign_j] += vci
                    modals[assign_i, assign_j] += 1

    return np.where(modals > 0, width / modals, -999)


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
    raster_specs = Raster.raster_specs(bankfull_width_src)
    csx, csy = raster_specs["csx"], raster_specs["csy"]

    valleys = da.ma.filled(from_raster(twi_src) >= twi_threshold, False).astype(bool)
    valley_eroded = binary_erosion(valleys, np.ones((1, 3, 3), dtype=bool)).astype(bool)
    valley_edges = valleys & ~valley_eroded

    regions, edges, bfw = da.compute(
        da.squeeze(da.where(valleys, 0, -999)).astype(np.float64),
        da.squeeze(valley_edges),
        da.squeeze(da.ma.filled(from_raster(bankfull_width_src), -999)).astype(
            np.float64
        ),
    )

    vci = da.from_array(
        vci_width_transform_task(regions, edges, bfw, csx, csy)[np.newaxis, :]
    )
    del regions, edges, bfw

    vci = da.ma.masked_where(vci == -999, vci)

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
        cutoff (Union[float, None]): Return the index up to a defined threshold.
        Defaults to None.
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
    **kwargs,
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
