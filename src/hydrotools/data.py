# Canada's Medium Resolution DEM (30 m DTM), read remotely as a COG
from pyproj import CRS

from hydrotools.raster import Raster


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

# Maximum number of cells used to screen the area draining into a DEM
SCREENING_CELLS = 4_000_000

# Avoid listing remote directories when opening remote (/vsicurl/) rasters
REMOTE_WARP_ARGS = ["--config", "GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"]


def _transform_geom(geom, in_proj, out_proj):
    """Transform a shapely geometry"""
    transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)

    return shapely.transform(
        geom, lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1]))
    )


def _hydrobasins_path(region: str) -> str:
    """Download (once) a HydroBASINS level 12 region into the cache directory

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
        urlretrieve(HYDROBASINS_URL.format(region=region), tmp_path)
        os.replace(tmp_path, zip_path)

    return f"zip://{zip_path}!{name}.shp"


def _hydrobasins_upstream(footprint) -> list:
    """Collect HydroBASINS level 12 basins that intersect a footprint, along with all
    basins upstream of them.

    Args:
        footprint (shapely.Geometry): Footprint in EPSG:4326.

    Returns:
        list: Basin geometries in EPSG:4326.
    """
    next_down = {}
    region_ids = {}
    seeds = set()

    for region in HYDROBASINS_REGIONS:
        path = _hydrobasins_path(region)

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

    return geoms


def containing_basins(src: str):
    r = Raster(src)
    src_crs = CRS.from_wkt(r.wkt)

    footprint = _transform_geom(
        shapely.segmentize(box(*r.bbox), max(r.right - r.left, r.top - r.bottom) / 20),
        src_crs,
        "EPSG:4326",
    )

    # An equal-area grid centred on the DEM is used for all regional analysis
    lon, lat = footprint.centroid.x, footprint.centroid.y
    laea = (
        f"+proj=laea +lat_0={lat:.6f} +lon_0={lon:.6f} +x_0=0 +y_0=0 +ellps=GRS80 "
        "+units=m +no_defs"
    )

    print("Collecting HydroBASINS upstream of the DEM")
    geoms = [
        _transform_geom(g, "EPSG:4326", laea)
        for g in _hydrobasins_upstream(footprint) + [footprint]
    ]
