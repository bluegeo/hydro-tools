import os
from types import SimpleNamespace
from tempfile import TemporaryDirectory, _get_candidate_names
from collections import OrderedDict

import numpy as np
import dask.array as da
import rasterio.mask
import fiona
from shapely import geometry
from rtree import Index

from numba import njit
from hydrotools.config import TMP_DIR
from hydrotools.utils import TempRasterFile, TempRasterFiles
from hydrotools.raster import (
    Raster,
    from_raster,
    to_raster,
    warp_like,
    vector_to_raster,
)
from hydrotools.watershed import (
    flow_direction_accumulation,
    extract_streams,
    FlowAccumulation,
    stream_order,
)
from hydrotools.elevation import slope, solar_radiation, terrain_ruggedness_index
from hydrotools.morphology import bankfull_width_geometric, stream_slope


@njit
def _ind_to_coord(i: int, j: int, top: float, left: float, csx: float, csy: float):
    x = left + (j + 0.5) * csx
    y = top - (i + 0.5) * csy

    return x, y


@njit
def _stream_topology_task(
    reaches: np.ndarray,
    fd: np.ndarray,
    nodata: float | int,
    top: float,
    left: float,
    csx: float,
    csy: float,
):
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
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

    complete = np.full(reaches.shape, False, dtype=bool)
    # Init reaches and stream ID's with dummy to set types
    output_reaches = [[(0.0, 0.0)]]
    reach_ids = [0]
    next_reach_ids = [0]

    rows, cols = reaches.shape

    for i in range(rows):
        for j in range(cols):
            if reaches[i, j] != nodata and not complete[i, j]:
                current_val = reaches[i, j]

                # Delineate this reach
                complete[i, j] = True
                reach = [_ind_to_coord(i, j, top, left, csx, csy)]
                reach_ids.append(current_val)

                # Upstream cells
                stack = [(i, j)]
                while len(stack) > 0:
                    ci, cj = stack.pop()

                    # Find the next upstream neighbour
                    for ni, nj in nbrs:
                        nbr_i, nbr_j = ci + ni, cj + nj

                        if (
                            reaches[nbr_i, nbr_j] != current_val
                            or complete[nbr_i, nbr_j]
                        ):
                            continue

                        # Downstream cell of this offset
                        nbr_fd = fd[nbr_i, nbr_j]

                        if nbr_fd > 0:
                            i_offset, j_offset = directions[nbr_fd]
                            test_i, test_j = nbr_i + i_offset, nbr_j + j_offset

                            # Check if the downstream cell is the current cell
                            if test_i == ci and test_j == cj:
                                complete[nbr_i, nbr_j] = True
                                stack.append((nbr_i, nbr_j))
                                reach.append(
                                    _ind_to_coord(nbr_i, nbr_j, top, left, csx, csy)
                                )

                # Reverse the upstream cells (upstream to downstream)
                # and add downstream cells
                reach = reach[::-1]

                stack = [(i, j)]
                while len(stack) > 0:
                    ci, cj = stack.pop()

                    current_fd = fd[ci, cj]

                    if current_fd > 0:
                        # Find next downstream neighbour
                        i_offset, j_offset = directions[current_fd]
                        test_i, test_j = ci + i_offset, cj + j_offset

                        if (
                            not complete[test_i, test_j]
                            and reaches[test_i, test_j] == current_val
                        ):
                            complete[test_i, test_j] = True
                            stack.append((test_i, test_j))
                            reach.append(
                                _ind_to_coord(test_i, test_j, top, left, csx, csy)
                            )

                # Extend the reach to include the next downstream cell to
                # ensure reaches are connected
                if reaches[test_i, test_j] != nodata and current_fd > 0:
                    reach.append(_ind_to_coord(test_i, test_j, top, left, csx, csy))
                    next_reach_ids.append(reaches[test_i, test_j])
                else:
                    next_reach_ids.append(-1)

                output_reaches.append(reach)

    return output_reaches[1:], reach_ids[1:], next_reach_ids[1:]


def _stream_topology(
    reaches_array: np.ndarray, nodata: float | int, fd: str, dst: str
) -> list[list[tuple[int, int]]]:
    fd_array = from_raster(fd)[0, ...].compute().filled(-1)

    fd_rast = Raster(fd)

    reaches, reach_ids, next_reach_ids = _stream_topology_task(
        reaches_array,
        fd_array,
        nodata,
        fd_rast.top,
        fd_rast.left,
        fd_rast.csx,
        fd_rast.csy,
    )

    reach_mapping = {}
    for reach_id, next_reach_id in zip(reach_ids, next_reach_ids):
        if next_reach_id == -1:
            continue
        reach_enum = 0
        while True:
            reach_enum += 1
            key = f"{next_reach_id}:{reach_enum}"

            try:
                reach_mapping[key]
            except KeyError:
                break

        reach_mapping[key] = reach_id

    num_prv_fields = max([int(f.split(":")[1]) for f in reach_mapping.keys()])

    stream_features = []
    for reach, reach_id, next_reach_id in zip(reaches, reach_ids, next_reach_ids):
        if len(reach) < 2:
            continue

        props = {"rid": reach_id, "next_rid": next_reach_id}

        for i in range(num_prv_fields):
            key = f"{reach_id}:{i + 1}"
            try:
                props[f"prev_rid0{i + 1}"] = reach_mapping[key]
            except KeyError:
                props[f"prev_rid0{i + 1}"] = -1

        stream_features.append(
            {
                "geometry": geometry.mapping(geometry.LineString(reach)),
                "properties": props,
            }
        )

    # Calculate strahler order for each reach.
    # Dangling reaches (those that flow into a single point) are also fixed here.
    reach_mapping = {reach["properties"]["rid"]: reach for reach in stream_features}

    # Start with all leaf nodes (furthest upstream segments)
    stack = [
        reach for reach in stream_features if reach["properties"]["prev_rid01"] == -1
    ]
    while len(stack) > 0:
        reach = stack.pop(0)

        # Determine the strahler order for this node
        try:
            upstream_strahlers = [
                reach_mapping[value]["properties"]["strahler"]
                for field, value in reach["properties"].items()
                if "prev_rid" in field and value != -1
            ]
        except KeyError:
            # Not all upstream reaches have been processed yet.
            # Save this one until later.
            stack.append(reach)
            continue

        if len(upstream_strahlers) == 0:
            strahler = 1
        else:
            max_strahler = max(upstream_strahlers)

            if sum([s == max_strahler for s in upstream_strahlers]) == 1:
                strahler = max_strahler
            else:
                strahler = max_strahler + 1

        reach["properties"]["strahler"] = strahler

        # Traverse to the next node
        while True:
            next_reach_id = reach["properties"]["next_rid"]

            if next_reach_id == -1:
                break

            try:
                reach = reach_mapping[next_reach_id]
            except KeyError:
                # This is the end - the downstream was a single point
                reach["properties"]["next_rid"] = -1
                break

            # Check if a new node (confluence)
            if reach["properties"]["prev_rid02"] != -1:
                if reach["properties"]["rid"] not in [
                    r["properties"]["rid"] for r in stack
                ]:
                    stack.append(reach)
                break

            reach["properties"]["strahler"] = strahler

    # Calculate geometric parameters and distances from outlets
    outlets = [
        reach for reach in stream_features if reach["properties"]["next_rid"] == -1
    ]

    net_id = 0
    for outlet in outlets:
        net_id += 1

        nodes = [(outlet, 0.0)]
        while len(nodes) > 0:
            reach, distance = nodes.pop(0)
            props = reach["properties"]

            line = geometry.shape(reach["geometry"])

            length = line.length
            sinuosity = (
                length / geometry.LineString([line.coords[0], line.coords[-1]]).length
            )
            distance += length

            props["netID"] = net_id
            props["length"] = length
            props["sinuosity"] = sinuosity
            props["upDist"] = distance

            for field, value in props.items():
                if "prev_rid" in field and value != -1:
                    nodes.append((reach_mapping[value], distance))

    with fiona.open(
        dst,
        "w",
        driver="GPKG",
        crs=fd_rast.wkt,
        schema={
            "geometry": "LineString",
            "properties": {
                "rid": "int",
                "next_rid": "int",
                **OrderedDict(
                    {f"prev_rid0{i + 1}": "int" for i in range(num_prv_fields)}
                ),
                "strahler": "int",
                "netID": "int",
                "length": "float",
                "sinuosity": "float",
                "upDist": "float",
            },
        },
    ) as layer:
        layer.writerecords(stream_features)


@njit
def _fa_label_task(reaches: np.ndarray, fd: np.ndarray, nodata: float | int):
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
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

    labels = np.full(reaches.shape, 0, dtype=np.uint32)

    rows, cols = reaches.shape

    current_label = 0
    for i in range(rows):
        for j in range(cols):
            if reaches[i, j] != nodata and labels[i, j] == 0:
                current_val = reaches[i, j]

                # Delineate this reach
                current_label += 1
                labels[i, j] = current_label
                stack = [(i, j)]

                while len(stack) > 0:
                    ci, cj = stack.pop()

                    current_fd = fd[ci, cj]

                    if current_fd > 0:
                        # Find next downstream neighbour
                        i_offset, j_offset = directions[current_fd]
                        test_i, test_j = ci + i_offset, cj + j_offset

                        if (
                            labels[test_i, test_j] == 0
                            and reaches[test_i, test_j] == current_val
                        ):
                            labels[test_i, test_j] = current_label
                            stack.append((test_i, test_j))

                    # Find the next upstream neighbour
                    for ni, nj in nbrs:
                        nbr_i, nbr_j = ci + ni, cj + nj

                        if (
                            reaches[nbr_i, nbr_j] != current_val
                            or labels[nbr_i, nbr_j] != 0
                        ):
                            continue

                        # Downstream cell of this offset
                        nbr_fd = fd[nbr_i, nbr_j]

                        if nbr_fd > 0:
                            i_offset, j_offset = directions[nbr_fd]
                            test_i, test_j = nbr_i + i_offset, nbr_j + j_offset

                            # Check if the downstream cell is the current cell
                            if test_i == ci and test_j == cj:
                                labels[nbr_i, nbr_j] = current_label
                                stack.append((nbr_i, nbr_j))

    return labels


def _fa_label(reaches_array: np.ndarray, nodata: float | int, fd: str, labels_dst: str):
    fd_array = from_raster(fd)[0, ...].compute().filled(-1)

    labels = _fa_label_task(reaches_array, fd_array, nodata)
    labels_dask = da.from_array(
        labels.reshape(1, *labels.shape), chunks=from_raster(fd).chunks
    )

    labels_dask = da.ma.masked_where(labels_dask == 0, labels_dask)

    to_raster(
        labels_dask,
        fd,
        labels_dst,
    )


class Feat(SimpleNamespace):
    def __init__(self, feat: fiona.Collection):
        try:
            self.geo = geometry.shape(feat.geometry)
        except:
            self.geo = None
        self.props = SimpleNamespace(**feat.properties)

    @property
    def fiona_record(self):
        return {
            "geometry": geometry.mapping(self.geo),
            "properties": self.props.__dict__,
        }


def tree_gen(feats):
    for i, feat in enumerate(feats):
        yield (i, feat.geo.bounds, None)


def attr_to_line(line, raster, array, method="mean"):
    shape_mask, _, window = rasterio.mask.raster_geometry_mask(
        raster, [line], crop=True
    )

    data = array[
        window.row_off : window.row_off + window.height,
        window.col_off : window.col_off + window.width,
    ]

    return getattr(np.ma.masked_where(shape_mask, data), method)()


def summarize_on_streams(
    fa: FlowAccumulation, streams: da.Array, src: str, dst: str, method: str = "mean"
):
    with TempRasterFile() as temp_raster:
        fa.calculate(src, temp_raster, method)
        to_raster(
            da.ma.masked_where(da.ma.getmaskarray(streams), from_raster(temp_raster)),
            src,
            dst,
        )


def stream_analysis(
    dem: str,
    precipitation: str,
    lakes: str,
    t_min: str,
    t_max: str,
    swe: str,
    landcover: str,
    landcover_mapping: dict,
    streams_dst: str,
    min_area: float = 1e6,
    min_length: float = 25,
    working_directory: str = TMP_DIR,
    **kwargs,
):
    """
    Run a watershed analysis by extracting streams from a DEM and applying
    attributes to stream reaches.

    Args:
        dem (str): Path to a Digital Elevation Model raster
        precipitation (str): Path to a precipitation raster
        lakes (str): Path to a vector file of lake polygons
        streams_dst (str): Path to output streams vector file
        min_area (float, optional): Minimum watershed area for streams. Defaults to 1e6.
        min_length (float, optional): Minimum stream length. Defaults to 25.

    Kwargs:
        precip_max_mm (float, optional): Maximum precipitation value for scaling. Defaults to 3000.
        bw_coeff (float, optional): Bankfull width coefficient. Defaults to 0.042.
        a_exp (float, optional): Area exponent for bankfull width. Defaults to 0.48.
        p_exp (float, optional): Precipitation exponent for bankfull width. Defaults to 0.74.
        gradient_focal_mean_dist (int, optional): Distance for focal mean in gradient calculation. Defaults to 25.
        gradient_break_interval (int, optional): Interval for gradient breaks. Defaults to 1.
        bfw_break_interval (int, optional): Interval for bankfull width breaks. Defaults to 2.
        geometric_bfw_max_cost (float, optional): Maximum cost for geometric bankfull width. Defaults to 0.8.
    """
    # Output Rasters that can be saved if desired
    rasters = SimpleNamespace(
        fd=os.path.join(working_directory, "fd.tif"),
        fa=os.path.join(working_directory, "fa.tif"),
        fa_scaled=os.path.join(working_directory, "fa_scaled.tif"),
        streams=os.path.join(working_directory, "streams.tif"),
        mean_elev=os.path.join(working_directory, "mean_elev.tif"),
        lakes_src=os.path.join(working_directory, "lakes.tif"),
        precip_src=os.path.join(working_directory, "precip.tif"),
        t_min=os.path.join(working_directory, "t_min.tif"),
        t_max=os.path.join(working_directory, "t_max.tif"),
        swe=os.path.join(working_directory, "swe.tif"),
        area=os.path.join(working_directory, "area.tif"),
        mean_precip=os.path.join(working_directory, "mean_precip.tif"),
        bfw=os.path.join(working_directory, "bfw.tif"),
        gradient=os.path.join(working_directory, "gradient.tif"),
        valley_width=os.path.join(working_directory, "valley_width.tif"),
        valley_conf=os.path.join(working_directory, "valley_conf.tif"),
        reaches=os.path.join(working_directory, "reaches.tif"),
        solrad=os.path.join(working_directory, "solrad.tif"),
        solrad_sum=os.path.join(working_directory, "solrad_sum.tif"),
        tri=os.path.join(working_directory, "tri.tif"),
        slope_dst=os.path.join(working_directory, "slope.tif"),
    )
    for lc_name in landcover_mapping.values():
        setattr(
            rasters,
            lc_name,
            os.path.join(working_directory, f"lc_{lc_name}.tif"),
        )

    print("Calculating Flow Direction and Accumulation...")
    with TempRasterFile() as fd_tmp:
        flow_direction_accumulation(
            dem,
            fd_tmp,
            rasters.fa,
        )

    fa_data = fa_data = from_raster(rasters.fa).astype("float32")

    # Weight flow accumulation by precipitation
    warp_like(precipitation, rasters.precip_src, dem, as_cog=False)

    precip = from_raster(rasters.precip_src)
    precip_max_mm = kwargs.get("precip_max_mm", 3000)
    precip_scaled = da.clip(precip, 0, precip_max_mm) / precip_max_mm

    fa_data *= precip_scaled

    vector_to_raster(
        lakes, dem, rasters.lakes_src, as_mask=True, all_touched=True, as_cog=False
    )

    lakes_a = ~da.ma.getmaskarray(from_raster(rasters.lakes_src))

    fa_data = da.where(lakes_a, 0, fa_data)

    fa_data = da.ma.masked_where(da.ma.getmaskarray(from_raster(rasters.fa)), fa_data)
    to_raster(fa_data, rasters.fa, rasters.fa_scaled, as_cog=False)

    print("Extracting Streams...")
    extract_streams(
        dem,
        rasters.fa_scaled,
        rasters.streams,
        rasters.fd,
        min_area=min_area,
        min_length=min_length,
    )

    # -------------------------
    # Bankfull width estimation
    # -------------------------
    print("Estimating Bankfull Width...")
    flow_accum = FlowAccumulation(rasters.fd)
    flow_accum.contributing_area(rasters.area)

    streams_a = from_raster(rasters.streams)

    summarize_on_streams(flow_accum, streams_a, rasters.precip_src, rasters.mean_precip)

    precip_a = from_raster(rasters.mean_precip) / 10  # mm -> cm

    ca_a = from_raster(rasters.area)

    bankfull_width = (
        kwargs.get("bw_coeff", 0.196)
        * (ca_a ** kwargs.get("a_exp", 0.280))
        * (precip_a ** kwargs.get("p_exp", 0.355))
    )

    to_raster(
        da.ma.masked_where(da.ma.getmaskarray(streams_a), bankfull_width),
        rasters.streams,
        rasters.bfw,
    )

    # -------------------------
    # Stream Gradient, Valley Width, & Confinement
    # -------------------------
    print("Calculating Valley Width and Confinement...")
    stream_slope(
        dem,
        rasters.streams,
        rasters.gradient,
        focal_mean_dist=kwargs.get("gradient_focal_mean_dist", 25),
    )

    with TempRasterFile() as slope_tmp:
        slope(dem, slope_tmp)

        bankfull_width_geometric(
            rasters.streams,
            slope_tmp,
            rasters.valley_width,
            max_cost=kwargs.get("geometric_bfw_max_cost", 0.8),
        )

        summarize_on_streams(flow_accum, streams_a, slope_tmp, rasters.slope_dst)

    vc = from_raster(rasters.valley_width) / from_raster(rasters.bfw)
    vc = da.where((vc < 0) | da.isnan(vc) | da.isinf(vc), 0, vc)
    vc = da.ma.masked_where(da.ma.getmaskarray(from_raster(rasters.streams)), vc)

    to_raster(
        vc,
        dem,
        rasters.valley_conf,
    )

    # -------------------------
    # Reach Break Classification
    # -------------------------
    print("Classifying Reach Breaks...")
    grad_a = from_raster(rasters.gradient)
    gradient_breaks = np.arange(0, 90, kwargs.get("gradient_break_interval", 2))
    grad_classes = da.digitize(grad_a, gradient_breaks) + 1

    bfw_data = from_raster(rasters.bfw)
    bfw_breaks = np.arange(0, 100, kwargs.get("bfw_break_interval", 2))
    bfw_classes = da.digitize(bfw_data, bfw_breaks) + 1

    # Combine the classes to create unique reaches
    reach_classes = (
        streams_a.astype("uint64") * 1000000
        + bfw_classes.astype("uint64") * 1000
        + grad_classes.astype("uint64") * 10
        + lakes_a.astype("uint64")
    )

    _fa_label(
        da.where(da.ma.getmaskarray(streams_a), 0, reach_classes).compute()[0, ...],
        0,
        rasters.fd,
        rasters.reaches,
    )

    # -------------------------
    # Additional Topographic Calculations
    # -------------------------
    print("Calculating Solar Radiation...")
    days = list(range(15, 365, 30))
    all_rads = []
    with TempRasterFiles(len(days)) as solrads:
        for day, sr in zip(days, solrads):
            solar_radiation(dem, day, 4, sr)

            all_rads.append(from_raster(sr)[0, ...])

        rt = from_raster(dem)

        with TempRasterFile() as mean_solrad:
            to_raster(
                da.dstack(all_rads)
                .mean(axis=2)
                .reshape(1, rt.shape[1], rt.shape[2])
                .rechunk(rt.chunks),
                dem,
                mean_solrad,
                as_cog=False,
            )

            summarize_on_streams(flow_accum, streams_a, mean_solrad, rasters.solrad)
            summarize_on_streams(
                flow_accum, streams_a, mean_solrad, rasters.solrad_sum, method="sum"
            )

    print("Calculating Terrain Ruggedness Index...")
    with TempRasterFile() as tri:
        terrain_ruggedness_index(dem, tri)

        summarize_on_streams(flow_accum, streams_a, tri, rasters.tri)

    # -------------------------
    # Other met parameters
    # -------------------------
    print("Summarizing Meteorological Parameters...")
    with TempRasterFiles(3) as (t_min_src, t_max_src, swe_src):
        warp_like(t_min, t_min_src, dem, as_cog=False)
        warp_like(t_max, t_max_src, dem, as_cog=False)
        warp_like(swe, swe_src, dem, as_cog=False)

        summarize_on_streams(flow_accum, streams_a, t_min_src, rasters.t_min)
        summarize_on_streams(flow_accum, streams_a, t_max_src, rasters.t_max)
        summarize_on_streams(flow_accum, streams_a, swe_src, rasters.swe)

    # -------------------------
    # Landcover
    # -------------------------
    print("Summarizing Landcover...")
    with TempRasterFile() as landcover_src:
        warp_like(landcover, landcover_src, dem, as_cog=False, resample_method="near")

        r = Raster(landcover_src)
        lc_a = from_raster(landcover_src)

        for lc_key, lc_name in landcover_mapping.items():
            with TempRasterFile() as dst:
                to_raster(
                    (lc_a == lc_key).astype("float32") * r.csx * r.csy,
                    landcover_src,
                    dst,
                    as_cog=False,
                )

                summarize_on_streams(
                    flow_accum,
                    streams_a,
                    dst,
                    getattr(rasters, lc_name),
                    method="sum",
                )

    # Summarize mean elevation on streams
    summarize_on_streams(flow_accum, streams_a, dem, rasters.mean_elev)

    # -------------------------
    # Stream topology, order, and attributes
    # -------------------------
    with TemporaryDirectory() as tmp_dir:
        topo_dst = os.path.join(tmp_dir, next(_get_candidate_names()) + ".gpkg")

        # Vectorize streams and calculate topology, strahler order, and geometric
        # attributes
        print("Calculating Stream Topology...")
        _stream_topology(
            da.where(
                da.ma.getmaskarray(streams_a),
                0,
                from_raster(rasters.reaches),
            ).compute()[0, ...],
            0,
            rasters.fd,
            topo_dst,
        )

        # Add stream attributes
        with fiona.open(topo_dst) as topo_layer:
            crs = topo_layer.crs
            schema = topo_layer.schema

            topo_feats = [
                feat
                for feat in [Feat(feat) for feat in topo_layer]
                if feat.geo is not None
            ]

            # Add watershed area
            area = from_raster(rasters.area)[0, ...].compute()
            rast = rasterio.open(rasters.area)
            print("Adding contributing area to stream reaches...")
            for feat in topo_feats:
                setattr(
                    feat.props,
                    "in_area_km2",
                    float(
                        attr_to_line(
                            feat.geo,
                            rast,
                            area,
                            method="min",
                        )
                    ),
                )
                schema["properties"]["in_area_km2"] = "float"
                setattr(
                    feat.props,
                    "out_area_km2",
                    float(
                        attr_to_line(
                            feat.geo,
                            rast,
                            area,
                            method="max",
                        )
                    ),
                )
                schema["properties"]["out_area_km2"] = "float"

            del area

            # Elevation
            elev = from_raster(dem)[0, ...].compute()
            rast = rasterio.open(dem)
            print("Adding elevation to stream reaches...")
            for feat in topo_feats:
                setattr(
                    feat.props,
                    "elev_min",
                    float(
                        attr_to_line(
                            feat.geo,
                            rast,
                            elev,
                            method="min",
                        )
                    ),
                )
                schema["properties"]["elev_min"] = "float"
                setattr(
                    feat.props,
                    "elev_max",
                    float(
                        attr_to_line(
                            feat.geo,
                            rast,
                            elev,
                            method="max",
                        )
                    ),
                )
                schema["properties"]["elev_max"] = "float"
                setattr(
                    feat.props,
                    "elev_mean",
                    float(
                        attr_to_line(
                            feat.geo,
                            rast,
                            elev,
                            method="mean",
                        )
                    ),
                )
                schema["properties"]["elev_mean"] = "float"

            del elev

            # Add attributes extracted from rasters
            attrs = {
                "gradient": rasters.gradient,
                "bankfull_width": rasters.bfw,
                "valley_width": rasters.valley_width,
                "confinement": rasters.valley_conf,
                "ws_mean_elev": rasters.mean_elev,
                "ws_mean_solrad": rasters.solrad,
                "ws_sum_solrad": rasters.solrad_sum,
                "ws_mean_tri": rasters.tri,
                "ws_mean_slope": rasters.slope_dst,
                "ws_mean_precip_mm": rasters.mean_precip,
                "ws_mean_tmin": rasters.t_min,
                "ws_mean_tmax": rasters.t_max,
                "ws_mean_swe": rasters.swe,
            }
            attrs.update(
                {lc_name: getattr(rasters, lc_name) for lc_name in landcover_mapping.values()}
            )
            for attr_name, attr_rast in attrs.items():
                print(f"Adding {attr_name} to stream reaches...")
                a = from_raster(attr_rast)[0, ...].compute()
                rast = rasterio.open(attr_rast)
                for feat in topo_feats:
                    setattr(
                        feat.props,
                        attr_name,
                        float(
                            attr_to_line(
                                feat.geo,
                                rast,
                                a,
                                method="mean",
                            )
                        ),
                    )

            del a

            for attr in attrs.keys():
                schema["properties"][attr] = "float"

            print("Writing Stream Reaches to File...")
            with fiona.open(
                streams_dst, "w", "GPKG", schema=schema, crs=crs
            ) as out_layer:
                out_layer.writerecords([feat.fiona_record for feat in topo_feats])
