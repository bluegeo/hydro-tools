from typing import Union
from collections import OrderedDict
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool

import fiona
from shapely.geometry import LineString, Point
import dask.array as da
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
    accumulation: str, precip: str, bankfull: str, min_area: float = 1e6
):
    """Estimate theoretical bankfull width using:

    bankfull_depth = 0.196 * (contrib_area[km2] ** 0.280) * (precip[cm] ** 0.355)

    Method from:

    Hall, J. E., D. M. Holzer, and T. J. Beechie. 2007. Predicting river floodplain and lateral channel migration
        for salmon habitat conservation. Journal of the American Water Resources Association 43:786–797

    Note, ensure the spatial reference of the rasters is a projected system in metres.

    For the best results, the average annual precipitation should be summarized for
    every position on the streams using the result of a `WatershedIndex` mean grid.

    Args:
        accumulation (str): Flow accumulation grid
        (generated using `watershed.flow_direction_accumulation`)
        precip (str): Mean annual precipitation grid in mm
        bankfull (str): Destination raster for bankfull grid
        min_area (float, optional): Minimum contributing area for streams in m.
        Defaults to 1E6.
    """
    ca_specs = Raster(accumulation)
    fa = from_raster(accumulation)
    contrib_area = fa.astype("float32") * (ca_specs.csx * ca_specs.csy / 1e6)
    contrib_area = da.ma.masked_where(contrib_area < min_area / 1e6, contrib_area)

    precip_cm = from_raster(precip).astype("float32") / 10

    bankfull_width = 0.196 * (contrib_area**0.280) * (precip_cm**0.355)

    to_raster(
        da.ma.masked_where(
            da.isinf(bankfull_width) | da.isnan(bankfull_width), bankfull_width
        ),
        precip,
        bankfull,
    )


def bankfull_mask(dem: str, bankfull_width: str):
    """Generate a mask around streams guided by a distance derived from bankfull width,
     which is calculated using `morphology.bankfull`.

    Args:
        dem (str): [description]
        bankfull_width (str): [description]
    """
    # Create distance transform to streams

    # Interpolate nodata for bankfull width

    # Mask areas where distance exceeds interpolated bankfull

    # return mask
    raise NotImplementedError("Method not complete")


def topographic_wetness(
    accumulation: str,
    slope: str,
    destination: str,
    min_area: float = 1e6,
    cutoff: Union[float, None] = None,
):
    """Calculate a topographic wetness index

    Args:
        accumulation (str): Flow accumulation grid
        (generated using `watershed.flow_direction_accumulation`)
        slope (str): Topographic slope in Degrees.
        destination (str): Output TWI grid
        min_area (float, optional): Minimum contributing area for simulated streams.
        Defaults to 1e6.
        cutoff (Union[float, None]): Return the index up to a defined threshold. Defaults to None.
    """
    ca_specs = Raster(accumulation)
    fa = from_raster(accumulation)

    streams = da.ma.masked_where(fa < min_area / (ca_specs.csx * ca_specs.csy), fa)

    # Generate cost surface (normalized slope)
    cost = da.clip(from_raster(slope), 0.0, 90.0) / 90.0
    # Make cost 0 where simulated streams exist
    cost = raster_where(da.ma.getmaskarray(streams), cost, 0)

    # Generate least cost surface
    with TempRasterFile() as streams_path, TempRasterFile() as cost_path, TempRasterFile() as cost_surface:
        to_raster(cost, slope, cost_path, False)
        to_raster(streams, slope, streams_path, False)

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

        to_raster(cs_inverted, slope, destination)


def riparian_connectivity(dem: str, connectivity_dst, min_ws_area: float, **kwargs):
    """Calculate regions of riparian connectivity surrounding streams

    Args:
        dem (str): Input Digital Elevation Model raster
    """
    with TempRasterFiles(6) as (
        accumulation_dst,
        direction_dst,
        stream_dst,
        slope_dst,
        cost_dst,
        lcp_dst,
    ):
        # Derive streams
        flow_direction_accumulation(
            dem,
            direction_dst,
            accumulation_dst,
            False,
            False,
        )

        extract_streams(
            dem,
            accumulation_dst,
            stream_dst,
            direction_dst,
            min_ws_area,
            kwargs.get("min_length", 0),
        )

        # Create a cost surface
        slope(dem, slope_dst, overviews=False)

        cost = raster_where(
            da.ma.getmaskarray(from_raster(stream_dst)),
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

        # Bankfull width
        # contrib_area = fa.astype("float32") * (ca_specs.csx * ca_specs.csy / 1e6)
        # contrib_area = da.ma.masked_where(contrib_area < min_area / 1e6, contrib_area)

        # precip_cm = from_raster(precip).astype("float32") / 10

        # bankfull_width = 0.196 * (contrib_area**0.280) * (precip_cm**0.355)

        # Cost (already distributed)
        # 1 - da.clip(raster_where(from_raster(connectivity_dst), 0, conn_threshold) / conn_threshold

        # Classify into 3 zones based on distribution
