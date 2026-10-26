"""Collection of external datasets that support hydrological analysis"""

import os
import shutil
from typing import Union
from urllib.request import Request, urlopen

import fiona
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio import features
import shapely
from shapely.geometry import shape

from hydrotools.config import CACHE_DIR
from hydrotools.raster import Raster, warp


# Canada's Medium Resolution DEM (30 m DTM), read remotely as a COG
MRDEM_DTM = (
    "/vsicurl/https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/"
    "mrdem-30-dtm.tif"
)

HYDROBASINS_URL = (
    "https://data.hydrosheds.org/file/HydroBASINS/standard/"
    "hybas_{region}_lev12_v1c.zip"
)

# HydroBASINS regions covering Canada and the transboundary basins that drain into it
HYDROBASINS_REGIONS = ("na", "ar")

# Avoid listing remote directories when opening remote (/vsicurl/) rasters
REMOTE_WARP_ARGS = ["--config", "GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"]

# Maximum number of cells read along each axis to create a raster footprint
FOOTPRINT_CELLS = 2048


def transform_geom(geom, in_proj, out_proj):
    """Transform a shapely geometry

    Args:
        geom (shapely.Geometry): Input geometry.
        in_proj: Input coordinate reference system.
        out_proj: Output coordinate reference system.

    Returns:
        shapely.Geometry: Transformed geometry.
    """
    transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)

    return shapely.transform(
        geom, lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1]))
    )


def native_cell_size(src: str) -> float:
    """Cell size of a raster in metres (approximate for geographic rasters)"""
    r = Raster(src)
    cs = min(r.csx, r.csy)

    return cs * 111_320.0 if CRS.from_wkt(r.wkt).is_geographic else cs


def valid_footprint(src: str, crs="EPSG:4326"):
    """Calculate the footprint of valid data in a raster. Large rasters are read at a
    reduced resolution, which may slightly expand the footprint.

    Args:
        src (str): Input raster.
        crs (optional): Output coordinate reference system. Defaults to "EPSG:4326".

    Returns:
        shapely.Geometry: Footprint of valid data.
    """
    with rasterio.open(src) as ds:
        scale = max(ds.height, ds.width) / FOOTPRINT_CELLS
        if scale > 1:
            out_shape = (
                max(1, int(round(ds.height / scale))),
                max(1, int(round(ds.width / scale))),
            )
            mask = ds.dataset_mask(out_shape=out_shape) > 0
            transform = ds.transform * ds.transform.scale(
                ds.width / out_shape[1], ds.height / out_shape[0]
            )
        else:
            mask = ds.dataset_mask() > 0
            transform = ds.transform

        src_crs = ds.crs

    if not mask.any():
        raise ValueError(f"{src} does not contain any valid data")

    footprint = shapely.union_all(
        [
            shape(geom)
            for geom, _ in features.shapes(
                mask.astype(np.uint8), mask=mask, transform=transform
            )
        ]
    )

    # Densify edges so the footprint follows curves when transformed
    bounds = footprint.bounds
    footprint = shapely.segmentize(
        footprint, max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 20
    )

    return transform_geom(footprint, src_crs, crs)


def hydrobasins_path(region: str) -> str:
    """Download (once) a HydroBASINS level 12 region into the cache directory

    HydroBASINS:
        Lehner, B., Grill G. (2013). Global river hydrography and network routing:
        baseline data and new approaches to study the world's large river systems.
        Hydrological Processes, 27(15): 2171-2186.

    Args:
        region (str): HydroBASINS region code, ex. "na"

    Returns:
        str: A fiona-readable path to the region shapefile
    """
    cache = os.path.join(CACHE_DIR, "hydrobasins")
    os.makedirs(cache, exist_ok=True)

    name = f"hybas_{region}_lev12_v1c"
    zip_path = os.path.join(cache, f"{name}.zip")

    if not os.path.isfile(zip_path):
        print(f"Downloading HydroBASINS region {region} to {cache}")
        tmp_path = f"{zip_path}.part"
        # The HydroSHEDS server rejects the default urllib user agent
        request = Request(
            HYDROBASINS_URL.format(region=region),
            headers={"User-Agent": "hydro-tools"},
        )
        with urlopen(request) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f)
        os.replace(tmp_path, zip_path)

    return f"zip://{zip_path}!{name}.shp"


def upstream_basins(
    footprint, dissolve: bool = True, regions: tuple = HYDROBASINS_REGIONS
) -> Union[list, "shapely.Geometry"]:
    """Collect HydroBASINS level 12 basins that intersect a footprint, along with all
    basins upstream of them. HydroBASINS is downloaded once and cached in
    `hydrotools.config.CACHE_DIR` (set with the `HYDROTOOLS_CACHE_DIR` environment
    variable).

    Note, the basins intersecting the footprint may extend downstream of it. The
    result is the area that may drain into the footprint, not the area that does.

    Args:
        footprint (shapely.Geometry): Footprint in EPSG:4326 (see `valid_footprint`).
        dissolve (bool, optional): Dissolve the basins into a single geometry.
        Defaults to True.
        regions (tuple, optional): HydroBASINS regions to search. Defaults to
        `HYDROBASINS_REGIONS`.

    Returns:
        Union[list, shapely.Geometry]: Dissolved basins, or a list of basin geometries
        when `dissolve` is False, in EPSG:4326.
    """
    next_down = {}
    region_ids = {}
    seeds = set()

    for region in regions:
        path = hydrobasins_path(region)

        with fiona.open(path, ignore_geometry=True) as src:
            for f in src:
                next_down[f.properties["HYBAS_ID"]] = f.properties["NEXT_DOWN"]
                region_ids[f.properties["HYBAS_ID"]] = path

        with fiona.open(path) as src:
            seeds.update(
                f.properties["HYBAS_ID"]
                for f in src.filter(bbox=footprint.bounds)
                if shape(f.geometry).intersects(footprint)
            )

    upstream = {}
    for hybas_id, down_id in next_down.items():
        upstream.setdefault(down_id, []).append(hybas_id)

    selected = set(seeds)
    stack = list(seeds)
    while stack:
        for up_id in upstream.get(stack.pop(), []):
            if up_id not in selected:
                selected.add(up_id)
                stack.append(up_id)

    geoms = []
    for path in set(region_ids[hybas_id] for hybas_id in selected):
        ids = [hybas_id for hybas_id in selected if region_ids[hybas_id] == path]
        with fiona.open(path) as src:
            for c in range(0, len(ids), 500):
                where = f"HYBAS_ID IN ({','.join(map(str, ids[c : c + 500]))})"
                geoms += [shape(f.geometry) for f in src.filter(where=where)]

    if dissolve:
        return shapely.union_all(geoms)

    return geoms


def collect_elevation(
    dst: str,
    bounds: tuple,
    cs: float,
    crs: str,
    sources: Union[str, list] = MRDEM_DTM,
    resample_method: Union[str, None] = None,
    as_cog: bool = True,
):
    """Collect elevation data (Canada's MRDEM by default) onto a grid

    Args:
        dst (str): Output raster.
        bounds (tuple): Output bounds in the form (left, bottom, right, top).
        cs (float): Output cell size.
        crs (str): Output coordinate reference system.
        sources (Union[str, list], optional): Elevation source(s) readable by GDAL.
        URLs are read remotely. Where sources overlap, later sources take priority.
        Defaults to MRDEM.
        resample_method (Union[str, None], optional): Resample method. Defaults to
        "average" when the output is coarser than the first source, otherwise
        "bilinear".
        as_cog (bool, optional): Output a COG. Defaults to True.
    """
    if isinstance(sources, str):
        sources = [sources]
    sources = [
        f"/vsicurl/{src}" if src.startswith(("http://", "https://")) else src
        for src in sources
    ]

    if resample_method is None:
        out_cs = cs * 111_320.0 if CRS.from_user_input(crs).is_geographic else cs
        resample_method = (
            "average" if out_cs > native_cell_size(sources[0]) else "bilinear"
        )

    warp(
        sources,
        dst,
        bounds,
        cs,
        cs,
        crs,
        "Float32",
        resample_method,
        additional_args=REMOTE_WARP_ARGS,
        as_cog=as_cog,
    )
