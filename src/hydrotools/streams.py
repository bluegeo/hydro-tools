import os
from types import SimpleNamespace
from tempfile import TemporaryDirectory, _get_candidate_names
from collections import OrderedDict

import numpy as np
import dask.array as da
import rasterio.mask
import fiona
from skimage.measure import label as image_label
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
def _fa_label_task(reaches: np.ndarray, fa: np.ndarray, nodata: float | int):
    labels = np.full(reaches.shape, -1, dtype=np.int32)

    rows, cols = reaches.shape

    current_label = 0
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(rows):
        for j in range(cols):
            if reaches[i, j] != nodata and labels[i, j] == -1:
                current_val = reaches[i, j]

                # Delineate this reach
                current_label += 1
                labels[i, j] = current_label
                stack = [(i, j)]

                while len(stack) > 0:
                    ci, cj = stack.pop()

                    current_fa = fa[ci, cj]

                    # Find next downstream neighbour
                    dif = 1000000
                    test_i, test_j = -1, -1
                    for di, dj in nbrs:
                        ni, nj = ci + di, cj + dj
                        if (
                            ni >= 0
                            and ni < rows
                            and nj >= 0
                            and nj < cols
                            and reaches[ni, nj] != nodata
                        ):
                            nd = abs(fa[ni, nj] - current_fa)
                            if fa[ni, nj] < current_fa and nd < dif:
                                dif = nd
                                test_i, test_j = ni, nj

                    if (
                        test_i != -1
                        and test_j != -1
                        and labels[test_i, test_j] == -1
                        and reaches[test_i, test_j] == current_val
                    ):
                        labels[test_i, test_j] = current_label
                        stack.append((test_i, test_j))

                    # Find the next upstream neighbour
                    dif = 1000000
                    test_i, test_j = -1, -1
                    for di, dj in nbrs:
                        ni, nj = ci + di, cj + dj
                        if (
                            ni >= 0
                            and ni < rows
                            and nj >= 0
                            and nj < cols
                            and reaches[ni, nj] != nodata
                        ):
                            nd = abs(fa[ni, nj] - current_fa)
                            if fa[ni, nj] > current_fa and nd < dif:
                                dif = nd
                                test_i, test_j = ni, nj

                    if (
                        test_i != -1
                        and test_j != -1
                        and labels[test_i, test_j] == -1
                        and reaches[test_i, test_j] == current_val
                    ):
                        labels[test_i, test_j] = current_label
                        stack.append((test_i, test_j))

    return labels


def _fa_label(reaches_array: np.ndarray, nodata: float | int, fa: str, labels_dst: str):
    fa_array = from_raster(fa)[0, ...].compute().filled(0)

    labels = _fa_label_task(reaches_array, fa_array, nodata)
    labels_dask = da.from_array(
        labels.reshape(1, *labels.shape), chunks=from_raster(fa).chunks
    )
    labels_dask = da.ma.masked_where(labels_dask == -1, labels_dask)

    to_raster(
        labels_dask,
        fa,
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
    gradient_breaks = np.arange(0, 90, kwargs.get("gradient_break_interval", 1))
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
        rasters.area,
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

    # -------------------------
    # Stream topology and order
    # -------------------------
    remove_attrs = [
        "cat",
        "horton",
        "shreve",
        "hack",
        "topo_dim",
        "scheidegger",
        "drwal_old",
        "flow_accum",
        "out_dist",
    ]
    attrs = {
        "avg_bfw": rasters.bfw,
        "avg_vw": rasters.valley_width,
        "avg_conf": rasters.valley_conf,
        "avg_solrad": rasters.solrad,
        "avg_tri": rasters.tri,
        "avg_slope": rasters.slope_dst,
        "avg_precip_mm": rasters.mean_precip,
        "avg_tmin": rasters.t_min,
        "avg_tmax": rasters.t_max,
        "avg_swe": rasters.swe,
    }
    attrs.update(
        {lc_name: getattr(rasters, lc_name) for lc_name in landcover_mapping.values()}
    )

    with TemporaryDirectory() as tmp_dir:
        topo_dst = os.path.join(tmp_dir, next(_get_candidate_names()) + ".gpkg")
        order_dst = os.path.join(tmp_dir, next(_get_candidate_names()) + ".gpkg")

        # First, add topology to reach breaks
        print("Calculating Stream Topology and Order...")
        stream_order(
            dem,
            rasters.reaches,
            rasters.fd,
            rasters.fa_scaled,
            topo_dst,
            only_topology=True,
        )

        # Proceed with normal Calculation of stream order on streams without breaks
        stream_order(
            dem,
            rasters.streams,
            rasters.fd,
            rasters.fa_scaled,
            order_dst,
        )

        # Vectorize and add stream attributes
        with fiona.open(topo_dst) as topo_layer, fiona.open(order_dst) as order_layer:
            crs = topo_layer.crs
            schema = topo_layer.schema
            max_prv = 0

            topo_feats = [
                feat
                for feat in [Feat(feat) for feat in topo_layer]
                if feat.geo is not None
            ]

            topo_tree = Index(tree_gen(topo_feats))

            order_feats = [
                feat
                for feat in [Feat(feat) for feat in order_layer]
                if feat.geo is not None
            ]

            order_tree = Index(tree_gen(order_feats))

            for root_feat in topo_feats:
                # Set the stream order attributes
                order_idx = order_tree.intersection(root_feat.geo.bounds)

                order_feat_ints = [
                    (order_feats[i], order_feats[i].geo.intersection(root_feat.geo))
                    for i in order_idx
                ]

                order_feat = max(
                    order_feat_ints,
                    key=lambda x: x[1].length if hasattr(x[1], "length") else -1,
                    default=None,
                )[0]

                root_feat.props.strahler = order_feat.props.strahler
                root_feat.props.horton = order_feat.props.horton
                root_feat.props.shreve = order_feat.props.shreve
                root_feat.props.hack = order_feat.props.hack

                # Set the previous stream attributes
                for field in list(root_feat.props.__dict__.keys()):
                    if "prev_str" in field:
                        delattr(root_feat.props, field)

                topo_idx = topo_tree.intersection(root_feat.geo.buffer(0.1).bounds)

                topo_proximal = [
                    topo_feats[i].props.stream
                    for i in topo_idx
                    if topo_feats[i].props.next_stream == root_feat.props.stream
                ]

                max_prv = max(max_prv, len(topo_proximal))

                for pidx, proximal in enumerate(topo_proximal):
                    setattr(root_feat.props, f"prev_str0{pidx + 1}", proximal)

            schema["properties"].update(
                OrderedDict([(f"prev_str0{i + 1}", "int") for i in range(max_prv)])
            )

            for feat in topo_feats:
                for i in range(max_prv):
                    if not hasattr(feat.props, f"prev_str0{i + 1}"):
                        setattr(feat.props, f"prev_str0{i + 1}", -1)

                # Spelling corrections
                setattr(feat.props, "straight", feat.props.stright)
                delattr(feat.props, "stright")

                # Remove unneeded fields
                for attr in remove_attrs:
                    delattr(feat.props, attr)

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

            # Add attributes extracted from rasters
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

            del schema["properties"]["stright"]
            schema["properties"]["straight"] = "float"
            for attr in remove_attrs:
                del schema["properties"][attr]
            for attr in attrs.keys():
                schema["properties"][attr] = "float"

            print("Writing Stream Reaches to File...")
            with fiona.open(
                streams_dst, "w", "GPKG", schema=schema, crs=crs
            ) as out_layer:
                out_layer.writerecords([feat.fiona_record for feat in topo_feats])
