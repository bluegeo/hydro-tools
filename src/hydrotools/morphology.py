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
    raster_where,
    from_raster,
    to_raster,
)
from hydrotools.utils import GrassRunner


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
            sinuosity_vector, "w", driver="GPKG", crs=vect.crs, schema=output_schema,
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

    bankfull_width = 0.196 * (contrib_area ** 0.280) * (precip_cm ** 0.355)

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
    pass


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
