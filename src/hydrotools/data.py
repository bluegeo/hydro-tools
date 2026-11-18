"""Collection of external datasets that support hydrological analysis"""

import os
import shutil
from logging import warning
from typing import Union
from urllib.request import Request, urlopen

import fiona
import numpy as np
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio import features
from shapely.geometry import shape

from hydrotools.config import CACHE_DIR
from hydrotools.raster import Raster, warp

# Canada's Medium Resolution DEM (30 m DTM), read remotely as a COG
MRDEM_DTM = (
    "/vsicurl/https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/"
    "mrdem-30-dtm.tif"
)

HYDROBASINS_URL = (
    "https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_{region}_lev12_v1c.zip"
)

# HydroBASINS regions covering Canada and the transboundary basins that drain into it
HYDROBASINS_REGIONS = ("na", "ar")

HYDRORIVERS_URL = (
    "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_{region}_shp.zip"
)

# Earth radius (m) used for cell areas on geographic grids
EARTH_RADIUS = 6_371_008.8

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

    footprint = shapely.union_all([
        shape(geom)
        for geom, _ in features.shapes(
            mask.astype(np.uint8), mask=mask, transform=transform
        )
    ])

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
        _download(HYDROBASINS_URL.format(region=region), zip_path)

    return f"zip://{zip_path}!{name}.shp"


def hydrorivers_path(region: str) -> str:
    """Download (once) a HydroRIVERS region into the cache directory

    HydroRIVERS:
        Lehner, B., Grill G. (2013). Global river hydrography and network routing:
        baseline data and new approaches to study the world's large river systems.
        Hydrological Processes, 27(15): 2171-2186.

    Args:
        region (str): HydroSHEDS region code, ex. "na"

    Returns:
        str: A fiona-readable path to the region shapefile
    """
    cache = os.path.join(CACHE_DIR, "hydrorivers")
    os.makedirs(cache, exist_ok=True)

    name = f"HydroRIVERS_v10_{region}"
    zip_path = os.path.join(cache, f"{name}_shp.zip")

    if not os.path.isfile(zip_path):
        print(f"Downloading HydroRIVERS region {region} to {cache}")
        _download(HYDRORIVERS_URL.format(region=region), zip_path)

    return f"zip://{zip_path}!{name}_shp/{name}.shp"


def _download(url: str, dst: str):
    """Download a file, replacing the destination only once complete"""
    tmp_path = f"{dst}.part"
    # The HydroSHEDS server rejects the default urllib user agent
    request = Request(url, headers={"User-Agent": "hydro-tools"})
    with urlopen(request) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp_path, dst)


def _fetch_features(path: str, field: str, ids: list) -> dict:
    """Features from a vector dataset with a field in a list of ids: {id: feature}"""
    out = {}
    ids = list(ids)
    with fiona.open(path) as src:
        for c in range(0, len(ids), 500):
            where = f"{field} IN ({','.join(map(str, ids[c : c + 500]))})"
            for f in src.filter(where=where):
                out[f.properties[field]] = f
    return out


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
    sources: str | list = MRDEM_DTM,
    resample_method: str | None = None,
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


def collect_contributing_elevation(
    dem: str,
    dst: str,
    sources: str | list = MRDEM_DTM,
    regions: tuple = HYDROBASINS_REGIONS,
    as_cog: bool = True,
):
    """Expand a DEM to cover the area that may drain into it. Elevation is collected
    (Canada's MRDEM by default) over the bounds of the HydroBASINS draining into the
    DEM (see `upstream_basins`), on the DEM grid. The DEM takes priority where it has
    valid data.

    Note, the output is the rectangular extent of the contributing basins, and may
    include area that does not drain into the DEM.

    Args:
        dem (str): Input DEM.
        dst (str): Output raster.
        sources (Union[str, list], optional): Elevation source(s) used outside the DEM
        (see `collect_elevation`). Defaults to MRDEM.
        regions (tuple, optional): HydroBASINS regions to search. Defaults to
        `HYDROBASINS_REGIONS`.
        as_cog (bool, optional): Output a COG. Defaults to True.
    """
    if isinstance(sources, str):
        sources = [sources]

    with rasterio.open(dem) as ds:
        crs = ds.crs
        csx, csy = ds.res
        left, top = ds.bounds.left, ds.bounds.top
    if csx != csy:
        raise ValueError("The DEM must have square cells")

    contributing = upstream_basins(valid_footprint(dem), regions=regions)
    c_left, c_bottom, c_right, c_top = transform_geom(
        contributing, "EPSG:4326", crs
    ).bounds

    # Snap outwards to the DEM grid so the DEM is not resampled
    bounds = (
        left + np.floor((c_left - left) / csx) * csx,
        top - np.ceil((top - c_bottom) / csy) * csy,
        left + np.ceil((c_right - left) / csx) * csx,
        top - np.floor((top - c_top) / csy) * csy,
    )

    collect_elevation(dst, bounds, csx, crs.to_wkt(), sources + [dem], as_cog=as_cog)


def line_crossings(line, footprint) -> list:
    """Find where a line, oriented in the direction of flow, crosses the boundary of
    a footprint

    Returns:
        list: (fraction along the line, True where the line enters the footprint)
    """
    crossing = line.intersection(footprint.boundary)
    if crossing.is_empty:
        return []

    points = []
    for part in getattr(crossing, "geoms", [crossing]):
        if part.geom_type == "Point":
            points.append(part)
        else:
            # Collinear sections: use both ends
            points += [part.boundary.geoms[0], part.boundary.geoms[-1]]

    fractions = sorted(set(line.project(p, normalized=True) for p in points))
    inside = [
        footprint.contains(line.interpolate((a + b) / 2, normalized=True))
        for a, b in zip([0.0] + fractions, fractions + [1.0])
    ]

    return [
        (f, inside[k + 1])
        for k, f in enumerate(fractions)
        if inside[k] != inside[k + 1]
    ]


def _cell_areas(ds, window) -> np.ndarray:
    """Cell areas (m2) for each row of a window"""
    if not ds.crs.is_geographic:
        return np.full(window.height, abs(ds.res[0] * ds.res[1]))

    transform = ds.window_transform(window)
    rows = np.arange(window.height) + 0.5
    lat = np.radians(transform.f + rows * transform.e)
    dlat = abs(np.radians(transform.e))
    dlon = abs(np.radians(transform.a))

    return (
        EARTH_RADIUS**2 * dlon * np.abs(np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2))
    )


def zonal_sums(raster: str, geoms: dict, exclude=None, window_size: int = 2048) -> dict:
    """Sum raster values multiplied by cell area within polygons, read in windows

    Args:
        raster (str): Raster to summarize.
        geoms (dict): Polygons in EPSG:4326 keyed by id.
        exclude (shapely.Geometry, optional): Area (EPSG:4326) to exclude. Defaults
        to None.
        window_size (int, optional): Raster window size in cells. Defaults to 2048.

    Returns:
        dict: {id: (sum of value * area, sum of area)} in m2. Polygons without
        valid cells are not included.
    """
    keys = list(geoms)
    sums = np.zeros(len(keys))
    areas = np.zeros(len(keys))

    with rasterio.open(raster) as ds:
        crs = ds.crs.to_wkt()
        polys = [transform_geom(geoms[k], "EPSG:4326", crs) for k in keys]
        tree = shapely.STRtree(polys)
        if exclude is not None:
            exclude = transform_geom(exclude, "EPSG:4326", crs)

        for row0 in range(0, ds.height, window_size):
            for col0 in range(0, ds.width, window_size):
                window = rasterio.windows.Window(
                    col0,
                    row0,
                    min(window_size, ds.width - col0),
                    min(window_size, ds.height - row0),
                )
                selected = tree.query(shapely.box(*ds.window_bounds(window)))
                if selected.size == 0:
                    continue

                transform = ds.window_transform(window)
                shape = (window.height, window.width)
                labels = features.rasterize(
                    [(polys[i], i + 1) for i in selected],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype="int32",
                )
                if exclude is not None:
                    labels[
                        features.geometry_mask([exclude], shape, transform, invert=True)
                    ] = 0

                a = ds.read(1, window=window, masked=True)
                valid = (labels > 0) & ~np.ma.getmaskarray(a) & ~np.isnan(a.data)
                if not valid.any():
                    continue

                cell_area = np.broadcast_to(_cell_areas(ds, window)[:, None], shape)
                idx = labels[valid] - 1
                sums += np.bincount(idx, (a.data[valid] * cell_area[valid]), len(keys))
                areas += np.bincount(idx, cell_area[valid], len(keys))

    return {k: (sums[i], areas[i]) for i, k in enumerate(keys) if areas[i] > 0}


def sample_lines(raster: str, lines: dict) -> dict:
    """Mean raster value along lines, sampled at vertices after densifying to the
    raster cell size

    Args:
        raster (str): Raster to sample.
        lines (dict): Lines in EPSG:4326 keyed by id.

    Returns:
        dict: {id: mean value}. Lines without valid samples are not included.
    """
    out = {}
    with rasterio.open(raster) as ds:
        crs = ds.crs.to_wkt()
        nodata = ds.nodata
        for key, line in lines.items():
            line = transform_geom(line, "EPSG:4326", crs)
            coords = shapely.get_coordinates(shapely.segmentize(line, min(ds.res)))
            values = np.array([v[0] for v in ds.sample(coords)], dtype="float64")
            valid = ~np.isnan(values)
            if nodata is not None:
                valid &= ~np.isclose(values, nodata)
            if valid.any():
                out[key] = float(values[valid].mean())

    return out


class Inflows:
    """Flow entering a footprint from the area that drains into it, using HydroRIVERS
    and HydroBASINS (see `hydrorivers_path` and `hydrobasins_path`).

    Each HydroRIVERS reach that enters the footprint is an inflow. Its upstream area
    is the reach attribute UPLAND_SKM, interpolated to the crossing using the reach
    catchment (CATCH_SKM). Where a river leaves the footprint and re-enters, the
    area upstream of the exit is subtracted, as it is already included within the
    footprint.

    Means of rasters (ex. precipitation) upstream of each inflow are calculated with
    zonal statistics on the HydroBASINS level 12 polygons that are entirely upstream
    of the inflow, and by sampling the raster along the upstream reaches in polygons
    that are only partially upstream (for example the polygon containing the inflow).

    Note, tributaries that are too small for HydroRIVERS (upstream areas less than
    around 10 km2) are not included, and HydroSHEDS data are of lower quality north
    of 60 degrees latitude.
    """

    def __init__(self, footprint, regions: tuple = HYDROBASINS_REGIONS):
        """
        Args:
            footprint (shapely.Geometry): Footprint in EPSG:4326 (see
            `valid_footprint`).
            regions (tuple, optional): HydroSHEDS regions to search. Defaults to
            `HYDROBASINS_REGIONS`.
        """
        self.footprint = footprint
        self.regions = regions
        self._means = {}

        self._load_reaches()
        self._resolve()

    def _load_reaches(self):
        """Read the reach network and the geometry of reaches crossing the footprint"""
        tables = {k: [] for k in ("id", "down", "hybas", "catch", "upland", "region")}
        self.crossing_geoms = {}
        self.region_paths = []

        for r, region in enumerate(self.regions):
            path = hydrorivers_path(region)
            self.region_paths.append(path)
            with fiona.open(path) as src:
                for f in src.filter(bbox=self.footprint.bounds):
                    geom = shape(f.geometry)
                    if geom.intersects(self.footprint.boundary):
                        self.crossing_geoms[f.properties["HYRIV_ID"]] = geom

            with fiona.open(path, ignore_geometry=True) as src:
                n = len(src)
                arrays = {
                    "id": np.empty(n, np.int64),
                    "down": np.empty(n, np.int64),
                    "hybas": np.empty(n, np.int64),
                    "catch": np.empty(n, np.float64),
                    "upland": np.empty(n, np.float64),
                }
                for k, f in enumerate(src):
                    p = f.properties
                    arrays["id"][k] = p["HYRIV_ID"]
                    arrays["down"][k] = p["NEXT_DOWN"]
                    arrays["hybas"][k] = p["HYBAS_L12"]
                    arrays["catch"][k] = p["CATCH_SKM"]
                    arrays["upland"][k] = p["UPLAND_SKM"]
            for k, a in arrays.items():
                tables[k].append(a)
            tables["region"].append(np.full(n, r, np.int64))

        self.reaches = {k: np.concatenate(v) for k, v in tables.items()}

        # Upstream reaches of each reach, as a CSR adjacency over reach indices
        ids = self.reaches["id"]
        self._index = {int(i): k for k, i in enumerate(ids)}
        order = np.argsort(self.reaches["down"], kind="stable")
        down_sorted = self.reaches["down"][order]
        self._up_start = np.searchsorted(down_sorted, ids, side="left")
        self._up_end = np.searchsorted(down_sorted, ids, side="right")
        self._up_order = order

        # Reaches in each HydroBASINS polygon
        self._hybas_ids, self._hybas_counts = np.unique(
            self.reaches["hybas"], return_counts=True
        )

    def _upstream(self, k: int) -> np.ndarray:
        """Indices of reaches flowing directly into reach index k"""
        return self._up_order[self._up_start[k] : self._up_end[k]]

    def _resolve(self):
        """Find inflows and the reaches upstream of each"""
        crossings = {}
        for hyriv_id, geom in self.crossing_geoms.items():
            found = line_crossings(geom, self.footprint)
            if found:
                crossings[self._index[hyriv_id]] = found

        def area_at(k, fraction):
            r = self.reaches
            return max(r["upland"][k] - r["catch"][k] * (1 - fraction), 0.0)

        self.entries = []
        for k, found in crossings.items():
            for n, (fraction, entering) in enumerate(found):
                if not entering:
                    continue

                area = area_at(k, fraction)
                subtracted = 0.0
                upstream = []
                if n > 0:
                    # The river leaves the footprint upstream on the same reach
                    subtracted += area_at(k, found[n - 1][0])
                else:
                    stack = list(self._upstream(k))
                    while stack:
                        u = stack.pop()
                        if u in crossings:
                            # Stop at the crossing nearest the end of the reach
                            f, entering_u = crossings[u][-1]
                            if not entering_u:
                                subtracted += area_at(u, f)
                            continue
                        upstream.append(u)
                        stack.extend(self._upstream(u))
                upstream = np.array(upstream, np.int64)

                # Polygons with every reach upstream of the inflow
                hybas, counts = np.unique(
                    self.reaches["hybas"][upstream], return_counts=True
                )
                total = self._hybas_counts[np.searchsorted(self._hybas_ids, hybas)]
                full = hybas[counts == total]
                partial = upstream[~np.isin(self.reaches["hybas"][upstream], full)]

                self.entries.append({
                    "reach": k,
                    "fraction": fraction,
                    "point": self
                    .crossing_geoms[int(self.reaches["id"][k])]
                    .interpolate(fraction, normalized=True)
                    .coords[0],
                    "area_km2": max(area - subtracted, 0.0),
                    "subtracted_km2": subtracted,
                    "upstream": upstream,
                    "polygons": full,
                    "partial": partial,
                })

    def __len__(self):
        return len(self.entries)

    @property
    def points(self) -> list:
        """Inflow locations (x, y) in EPSG:4326"""
        return [e["point"] for e in self.entries]

    @property
    def area(self) -> np.ndarray:
        """Area (m2) flowing in at each inflow"""
        return np.array([e["area_km2"] for e in self.entries]) * 1e6

    def _polygons(self, ids: np.ndarray) -> dict:
        """HydroBASINS polygons by id: {id: (geometry, SUB_AREA km2)}"""
        out = {}
        remaining = set(int(i) for i in ids)
        for region in self.regions:
            if not remaining:
                break
            found = _fetch_features(hydrobasins_path(region), "HYBAS_ID", remaining)
            out.update({
                i: (shape(f.geometry), f.properties["SUB_AREA"])
                for i, f in found.items()
            })
            remaining -= set(found)
        return out

    def _lines(self, indices: np.ndarray) -> dict:
        """HydroRIVERS reach geometries by reach index"""
        out = {}
        for r, path in enumerate(self.region_paths):
            sel = indices[self.reaches["region"][indices] == r]
            if sel.size == 0:
                continue
            ids = self.reaches["id"][sel]
            found = _fetch_features(path, "HYRIV_ID", ids.tolist())
            out.update({
                int(k): shape(found[int(i)].geometry) for k, i in zip(sel, ids)
            })
        return out

    def mean(self, raster: str) -> np.ndarray:
        """Mean value of a raster upstream of each inflow, outside of the footprint

        Args:
            raster (str): Raster readable by GDAL.

        Returns:
            np.ndarray: Mean value at each inflow. NaN where the raster does not
            cover the area upstream of an inflow.
        """
        if raster in self._means:
            return self._means[raster]

        if not self.entries:
            return np.zeros(0)

        polygon_ids = np.unique(np.concatenate([e["polygons"] for e in self.entries]))
        polygons = self._polygons(polygon_ids)
        poly_sums = zonal_sums(
            raster, {i: g for i, (g, _) in polygons.items()}, exclude=self.footprint
        )
        # Polygons smaller than the raster cells are sampled at a point
        unresolved = {
            i: polygons[i][0].representative_point()
            for i in polygons
            if i not in poly_sums
        }
        for i, value in sample_lines(raster, unresolved).items():
            area = polygons[i][1] * 1e6
            poly_sums[i] = (value * area, area)

        reach_idx = np.unique(
            np.concatenate(
                [e["partial"] for e in self.entries]
                + [np.array([e["reach"] for e in self.entries], np.int64)]
            )
        )
        lines = {k: g for k, g in self._lines(reach_idx).items()}
        line_means = sample_lines(raster, lines)

        means = np.full(len(self.entries), np.nan)
        catch = self.reaches["catch"]
        for n, e in enumerate(self.entries):
            total = 0.0
            area = 0.0
            for i in e["polygons"]:
                if i in poly_sums:
                    total += poly_sums[i][0]
                    area += poly_sums[i][1]
            for k in e["partial"]:
                if int(k) in line_means:
                    total += line_means[int(k)] * catch[k] * 1e6
                    area += catch[k] * 1e6
            if e["reach"] in line_means:
                w = catch[e["reach"]] * e["fraction"] * 1e6
                total += line_means[e["reach"]] * w
                area += w
            if area > 0:
                means[n] = total / area

        if np.isnan(means).any():
            warning(
                f"{raster} does not cover the area upstream of {np.isnan(means).sum()} "
                "inflow(s)"
            )

        self._means[raster] = means
        return means

    def records(self) -> list:
        """Inflow attributes for review"""
        r = self.reaches
        return [
            {
                "x": e["point"][0],
                "y": e["point"][1],
                "hyriv_id": int(r["id"][e["reach"]]),
                "area_km2": float(e["area_km2"]),
                "subtracted_km2": float(e["subtracted_km2"]),
                "upstream_reaches": int(e["upstream"].size),
                "upstream_polygons": int(e["polygons"].size),
            }
            for e in self.entries
        ]
