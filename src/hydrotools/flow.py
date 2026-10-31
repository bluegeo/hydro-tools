"""Memory-bounded flow accumulation and stream extraction on D8 flow direction grids

Flow accumulation follows the tiled method of Barnes (2017), "Parallel non-divergent
flow accumulation for trillion cell digital elevation models on desktops or clusters",
Environmental Modelling & Software 92. The grid is processed in tiles, with numba
kernels applied to each tile:

1. Each tile is accumulated independently, treating flow from other tiles as unknown.
   The accumulation at each perimeter cell and the perimeter cell through which its
   flow leaves the tile are retained.
2. A graph of all perimeter cells is accumulated to find the flow arriving at each
   tile from other tiles. The graph holds `4 / tile_size` of the cells in the grid.
3. Each tile is accumulated again with the arriving flow added, and written.

Memory is bounded by the tile size and the number of tiles processed concurrently,
plus the perimeter graph. Point seeds (for example flow arriving from outside of the
grid) are added in the same way as flow arriving from other tiles.

Flow direction uses the GRASS r.watershed encoding (1-8 counterclockwise from
north-east). Values outside of 1-8 and nodata do not flow.
"""

from concurrent.futures import ThreadPoolExecutor
from logging import warning
import os
import threading
from typing import Union

import numpy as np
from numba import njit
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from hydrotools.raster import Raster
from hydrotools.utils import infer_nodata


# Row/col offsets indexed by r.watershed flow direction (1-8 counterclockwise from NE)
FD_OFF_I = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int64)
FD_OFF_J = np.array([0, 1, 0, -1, -1, -1, 0, 1, 1], dtype=np.int64)

# Approximate working memory per tile cell (bytes), excluding weight bands
TILE_BYTES_PER_CELL = 16

# Approximate working memory per tile cell for each weight band (bytes)
TILE_BYTES_PER_BAND = 16


@njit(nogil=True, cache=True)
def _perimeter_count(h, w):
    """Number of perimeter cells in a tile"""
    if h == 1:
        return w
    if w == 1:
        return h
    return 2 * w + 2 * (h - 2)


@njit(nogil=True, cache=True)
def _perimeter_index(h, w, i, j):
    """Index of a cell in the perimeter ordering (top row, bottom row, left column,
    right column), or -1 for interior cells"""
    if i == 0:
        return j
    if i == h - 1:
        return w + j
    if j == 0:
        return 2 * w + i - 1
    if j == w - 1:
        return 2 * w + h - 2 + i - 1
    return -1


@njit(nogil=True, cache=True)
def _perimeter_cells(h, w):
    """Row and column of each perimeter cell, in the perimeter ordering"""
    n = _perimeter_count(h, w)
    rows = np.empty(n, np.int64)
    cols = np.empty(n, np.int64)
    for p in range(n):
        if h == 1:
            rows[p] = 0
            cols[p] = p
        elif w == 1:
            rows[p] = 0 if p == 0 else (h - 1 if p == 1 else p - 1)
            cols[p] = 0
        elif p < w:
            rows[p] = 0
            cols[p] = p
        elif p < 2 * w:
            rows[p] = h - 1
            cols[p] = p - w
        elif p < 2 * w + h - 2:
            rows[p] = p - 2 * w + 1
            cols[p] = 0
        else:
            rows[p] = p - 2 * w - (h - 2) + 1
            cols[p] = w - 1
    return rows, cols


@njit(nogil=True, cache=True)
def _tile_accumulate(fd, acc, entry_i, entry_j, entry_vals):
    """Accumulate weights within a tile

    Args:
        fd: Flow direction (h, w), 1-8 where flow exists
        acc: Weights (bands, h, w), accumulated in place
        entry_i, entry_j: Cells receiving flow from elsewhere
        entry_vals: Flow received at each entry cell (bands, entries)

    Returns:
        int: Number of cells processed. Fewer than the number of cells in the tile
        indicates a cycle in the flow direction.
    """
    h, w = fd.shape
    bands = acc.shape[0]

    for n in range(entry_i.shape[0]):
        for k in range(bands):
            acc[k, entry_i[n], entry_j[n]] += entry_vals[k, n]

    indeg = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            d = fd[i, j]
            if d < 1 or d > 8:
                continue
            ni = i + FD_OFF_I[d]
            nj = j + FD_OFF_J[d]
            if ni >= 0 and ni < h and nj >= 0 and nj < w:
                indeg[ni, nj] += 1

    queue = np.empty(h * w, np.int32)
    tail = 0
    for i in range(h):
        for j in range(w):
            if indeg[i, j] == 0:
                queue[tail] = i * w + j
                tail += 1

    head = 0
    while head < tail:
        idx = queue[head]
        head += 1
        i = idx // w
        j = idx % w
        d = fd[i, j]
        if d < 1 or d > 8:
            continue
        ni = i + FD_OFF_I[d]
        nj = j + FD_OFF_J[d]
        if ni >= 0 and ni < h and nj >= 0 and nj < w:
            for k in range(bands):
                acc[k, ni, nj] += acc[k, i, j]
            indeg[ni, nj] -= 1
            if indeg[ni, nj] == 0:
                queue[tail] = ni * w + nj
                tail += 1

    return tail


@njit(nogil=True, cache=True)
def _tile_exits(fd, cells_i, cells_j):
    """Find the cell through which flow from each cell leaves a tile

    Returns:
        np.ndarray: Flat index (i * w + j) of the exit cell for each cell, or -1 where
        flow ends within the tile
    """
    h, w = fd.shape
    memo = np.full(h * w, -2, np.int64)
    stack = np.empty(h * w, np.int64)
    out = np.empty(cells_i.shape[0], np.int64)

    for n in range(cells_i.shape[0]):
        i = cells_i[n]
        j = cells_j[n]
        depth = 0
        result = -1
        while True:
            idx = i * w + j
            if memo[idx] != -2:
                result = memo[idx]
                break
            d = fd[i, j]
            if d < 1 or d > 8 or depth == h * w:
                result = -1
                break
            ni = i + FD_OFF_I[d]
            nj = j + FD_OFF_J[d]
            if ni < 0 or ni >= h or nj < 0 or nj >= w:
                result = idx
                break
            stack[depth] = idx
            depth += 1
            i = ni
            j = nj

        for s in range(depth):
            memo[stack[s]] = result
        memo[i * w + j] = result
        out[n] = result

    return out


@njit(nogil=True, cache=True)
def _tile_links(
    fd, exits, per_i, per_j, row0, col0, grid_h, grid_w, tile_size, tile_base, base
):
    """Link the perimeter cells of a tile into the perimeter graph

    Returns:
        tuple: Downstream node of each perimeter cell (-1 where none), and whether
        the link crosses into another tile
    """
    h, w = fd.shape
    n = per_i.shape[0]
    tiles_j = (grid_w + tile_size - 1) // tile_size
    down = np.full(n, -1, np.int64)
    inter = np.zeros(n, np.bool_)

    for p in range(n):
        e = exits[p]
        if e < 0:
            continue
        i = per_i[p]
        j = per_j[p]
        if e == i * w + j:
            # Flow leaves the tile from this cell
            d = fd[i, j]
            gi = row0 + i + FD_OFF_I[d]
            gj = col0 + j + FD_OFF_J[d]
            if gi < 0 or gi >= grid_h or gj < 0 or gj >= grid_w:
                continue
            ti = gi // tile_size
            tj = gj // tile_size
            th = min(tile_size, grid_h - ti * tile_size)
            tw = min(tile_size, grid_w - tj * tile_size)
            down[p] = tile_base[ti * tiles_j + tj] + _perimeter_index(
                th, tw, gi - ti * tile_size, gj - tj * tile_size
            )
            inter[p] = True
        else:
            down[p] = base + _perimeter_index(h, w, e // w, e % w)

    return down, inter


@njit(nogil=True, cache=True)
def _graph_accumulate(local, down, inter, ext_tot, ext_in):
    """Accumulate the perimeter graph

    Args:
        local: Accumulation within each node's tile (bands, nodes)
        down: Downstream node of each node, or -1
        inter: Whether each node's link crosses into another tile
        ext_tot: Flow arriving at each node from other tiles and seeds, including flow
        routed through the node's tile (bands, nodes). Seeds should be added before
        calling. Updated in place.
        ext_in: Flow arriving at each node from other tiles only (bands, nodes).
        Updated in place.

    Returns:
        int: Number of nodes processed
    """
    bands, n = local.shape
    indeg = np.zeros(n, np.int64)
    for p in range(n):
        if down[p] >= 0:
            indeg[down[p]] += 1

    queue = np.empty(n, np.int64)
    tail = 0
    for p in range(n):
        if indeg[p] == 0:
            queue[tail] = p
            tail += 1

    head = 0
    while head < tail:
        p = queue[head]
        head += 1
        t = down[p]
        if t < 0:
            continue
        if inter[p]:
            for k in range(bands):
                v = ext_tot[k, p] + local[k, p]
                ext_in[k, t] += v
                ext_tot[k, t] += v
        else:
            for k in range(bands):
                ext_tot[k, t] += ext_tot[k, p]
        indeg[t] -= 1
        if indeg[t] == 0:
            queue[tail] = t
            tail += 1

    return tail


def _normalize_weights(weights: list) -> list:
    """Weights as (raster path or None, scale) pairs"""
    out = []
    for weight in weights:
        if isinstance(weight, (int, float)):
            out.append((None, float(weight)))
        elif isinstance(weight, str):
            out.append((weight, 1.0))
        else:
            path, scale = weight
            out.append((path, float(scale)))
    return out


def tile_windows(height: int, width: int, tile_size: int) -> list:
    """Windows covering a grid, in tile order"""
    windows = []
    for row0 in range(0, height, tile_size):
        for col0 in range(0, width, tile_size):
            windows.append(
                Window(
                    col0, row0, min(tile_size, width - col0), min(tile_size, height - row0)
                )
            )
    return windows


class TiledAccumulation:
    """Flow accumulation of weight rasters over a D8 flow direction grid, processed in
    tiles with bounded memory. See the module docstring.

    Example, contributing area (m2) and upstream precipitation (mm m2) in one pass:

        acc = TiledAccumulation(direction)
        acc.accumulate(
            [cell_area, (precip, cell_area)], [area_dst, precip_dst], dtype="float64"
        )

    Weights are a list of any of the following, each producing an output band:

    * A number: a constant weight for every valid cell
    * A raster path: the raster values (warped to the direction grid as needed)
    * A tuple (raster path, scale): the raster values multiplied by a constant

    Seeds are a list of (row, col, values) that add flow at a cell, where values has
    an entry for every weight.
    """

    def __init__(
        self,
        direction: str,
        tile_size: int = 2048,
        workers: Union[int, None] = None,
        max_memory: int = 4096,
    ):
        """
        Args:
            direction (str): Flow direction raster (r.watershed encoding)
            tile_size (int, optional): Tile size in cells. A multiple of the raster
            block size (512) avoids rewriting blocks. Defaults to 2048.
            workers (Union[int, None], optional): Tiles processed concurrently.
            Defaults to the number of CPUs, limited by `max_memory`.
            max_memory (int, optional): Approximate limit (MB) on the memory used by
            tiles. The tile size is reduced when a single tile exceeds it. Defaults to
            4096.
        """
        self.direction = direction
        self.tile_size = int(tile_size)
        self.workers = workers if workers is not None else max(1, os.cpu_count() or 1)
        self.max_memory = max_memory

        with rasterio.open(direction) as ds:
            self.height = ds.height
            self.width = ds.width
            self.transform = ds.transform
            self.crs = ds.crs

        self._pass1 = None

    def _tiles(self, tile_size: int) -> tuple:
        """Windows, perimeter node offsets, and total nodes for a tile size"""
        windows = tile_windows(self.height, self.width, tile_size)
        counts = [_perimeter_count(w.height, w.width) for w in windows]
        base = np.zeros(len(windows), np.int64)
        base[1:] = np.cumsum(counts[:-1])
        return windows, base, int(sum(counts))

    def _plan(self, bands: int) -> tuple:
        """Tile size and workers that satisfy the memory limit"""
        per_cell = TILE_BYTES_PER_CELL + TILE_BYTES_PER_BAND * bands
        limit = self.max_memory * 2**20
        tile_size = self.tile_size
        while tile_size > 512 and tile_size**2 * per_cell > limit:
            tile_size -= 512
        workers = max(1, min(self.workers, limit // (tile_size**2 * per_cell)))
        return tile_size, workers

    def _open(self, path: str, warped: bool = False):
        """Open a raster, warped to the direction grid when needed. Handles are opened
        per tile, as they are not safe to share between threads."""
        ds = rasterio.open(path)
        if not warped:
            return ds
        return WarpedVRT(
            ds,
            crs=self.crs,
            transform=self.transform,
            width=self.width,
            height=self.height,
            resampling=Resampling.bilinear,
        )

    def _read_direction(self, window: Window) -> tuple:
        """Flow direction (1-8, 0 where none) and the valid data mask for a window"""
        with self._open(self.direction) as ds:
            a = ds.read(1, window=window, masked=True)
        valid = ~np.ma.getmaskarray(a)
        fd = a.filled(0)
        fd = np.where((fd < 1) | (fd > 8), 0, fd).astype(np.int8)
        return fd, valid

    def _read_weights(self, weights: list, window: Window, valid: np.ndarray) -> tuple:
        """Weight bands for a window, and the number of valid cells missing values"""
        acc = np.zeros((len(weights), window.height, window.width), np.float64)
        missing = 0
        for k, (path, scale) in enumerate(weights):
            if path is None:
                acc[k] = valid * scale
                continue
            with self._open(path, warped=not self._aligned[path]) as ds:
                a = ds.read(1, window=window, masked=True)
            a = np.ma.masked_invalid(a)
            mask = np.ma.getmaskarray(a)
            missing += int((mask & valid).sum())
            acc[k] = np.where(valid & ~mask, a.filled(0), 0) * scale
        return acc, missing

    def _check_alignment(self, weights: list):
        self._aligned = {}
        for path, _ in weights:
            if path is not None and path not in self._aligned:
                self._aligned[path] = Raster(path).matches(self.direction)

    def _run_tiles(self, fn, windows: list, workers: int):
        """Apply a tile function to windows concurrently"""
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(fn, range(len(windows))))

    def _first_pass(self, weights: list, tile_size: int, workers: int):
        """Accumulate each tile independently and link the perimeter graph"""
        windows, base, nodes = self._tiles(tile_size)
        bands = len(weights)
        local = np.zeros((bands, nodes), np.float64)
        down = np.full(nodes, -1, np.int64)
        inter = np.zeros(nodes, np.bool_)
        stats = {"missing": 0, "cycles": 0}
        lock = threading.Lock()

        def tile(t):
            window = windows[t]
            fd, valid = self._read_direction(window)
            acc, missing = self._read_weights(weights, window, valid)
            processed = _tile_accumulate(
                fd, acc, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((bands, 0))
            )
            per_i, per_j = _perimeter_cells(window.height, window.width)
            exits = _tile_exits(fd, per_i, per_j)
            t_down, t_inter = _tile_links(
                fd,
                exits,
                per_i,
                per_j,
                window.row_off,
                window.col_off,
                self.height,
                self.width,
                tile_size,
                base,
                base[t],
            )
            sl = slice(base[t], base[t] + per_i.shape[0])
            local[:, sl] = acc[:, per_i, per_j]
            down[sl] = t_down
            inter[sl] = t_inter
            with lock:
                stats["missing"] += missing
                stats["cycles"] += fd.size - processed

        self._run_tiles(tile, windows, workers)

        if stats["cycles"]:
            warning(f"{stats['cycles']} cells are part of flow direction cycles")

        return {
            "weights": weights,
            "tile_size": tile_size,
            "windows": windows,
            "base": base,
            "local": local,
            "down": down,
            "inter": inter,
            "missing": stats["missing"],
        }

    def _seed_tiles(self, seeds: list, tile_size: int) -> dict:
        """Group seeds by tile: {tile: (rows, cols, values)}"""
        tiles_j = (self.width + tile_size - 1) // tile_size
        grouped = {}
        for row, col, values in seeds:
            if not (0 <= row < self.height and 0 <= col < self.width):
                raise ValueError(f"Seed ({row}, {col}) is outside of the grid")
            t = (row // tile_size) * tiles_j + col // tile_size
            grouped.setdefault(t, []).append((row, col, values))

        out = {}
        for t, items in grouped.items():
            rows = np.array([s[0] for s in items], np.int64)
            cols = np.array([s[1] for s in items], np.int64)
            values = np.array([s[2] for s in items], np.float64).T
            out[t] = (rows, cols, values)
        return out

    def accumulate(
        self,
        weights: list,
        dst: list,
        seeds: Union[list, None] = None,
        dtype: str = "float32",
        nodata: Union[float, None] = None,
    ):
        """Accumulate weights and write the results

        Args:
            weights (list): Weights, one per output band (see the class docstring).
            dst (list): Output raster path for each weight.
            seeds (Union[list, None], optional): Flow added at cells, as
            (row, col, values) with a value for each weight. Defaults to None.
            dtype (str, optional): Output data type. Defaults to "float32".
            nodata (Union[float, None], optional): Output nodata value. Defaults to
            the minimum of the data type.
        """
        weights = _normalize_weights(weights)
        if len(dst) != len(weights):
            raise ValueError("An output path is required for every weight")
        bands = len(weights)
        tile_size, workers = self._plan(bands)
        self._check_alignment(weights)

        # The first pass depends only on the weights, so it is reused when seeds change
        if (
            self._pass1 is None
            or self._pass1["weights"] != weights
            or self._pass1["tile_size"] != tile_size
        ):
            print(f"Accumulating {bands} band(s) in {tile_size} cell tiles")
            self._pass1 = self._first_pass(weights, tile_size, workers)
            if self._pass1["missing"]:
                warning(
                    f"{self._pass1['missing']} valid cells have no weight values and "
                    "were given a weight of 0"
                )
        p1 = self._pass1
        windows, base = p1["windows"], p1["base"]
        nodes = p1["down"].shape[0]

        # Seeds are routed to the exit of their tile before the graph is accumulated
        seed_tiles = self._seed_tiles(seeds or [], tile_size)
        ext_tot = np.zeros((bands, nodes), np.float64)
        ext_in = np.zeros((bands, nodes), np.float64)
        for t, (rows, cols, values) in seed_tiles.items():
            window = windows[t]
            fd, _ = self._read_direction(window)
            exits = _tile_exits(fd, rows - window.row_off, cols - window.col_off)
            for n, e in enumerate(exits):
                if e >= 0:
                    node = base[t] + _perimeter_index(
                        window.height, window.width, e // window.width, e % window.width
                    )
                    ext_tot[:, node] += values[:, n]

        _graph_accumulate(p1["local"], p1["down"], p1["inter"], ext_tot, ext_in)

        if nodata is None:
            nodata = infer_nodata(dtype)
        for path in dst:
            Raster.empty_like(path, self.direction, dtype=dtype, nodata=nodata)
        out = [rasterio.open(path, "r+") for path in dst]
        lock = threading.Lock()

        def tile(t):
            window = windows[t]
            fd, valid = self._read_direction(window)
            acc, _ = self._read_weights(weights, window, valid)

            per_i, per_j = _perimeter_cells(window.height, window.width)
            sl = slice(base[t], base[t] + per_i.shape[0])
            entries = np.nonzero(np.any(ext_in[:, sl] != 0, axis=0))[0]
            entry_i = per_i[entries]
            entry_j = per_j[entries]
            entry_vals = ext_in[:, sl][:, entries]
            if t in seed_tiles:
                rows, cols, values = seed_tiles[t]
                entry_i = np.concatenate([entry_i, rows - window.row_off])
                entry_j = np.concatenate([entry_j, cols - window.col_off])
                entry_vals = np.concatenate([entry_vals, values], axis=1)

            _tile_accumulate(fd, acc, entry_i, entry_j, entry_vals)

            with lock:
                for k, ds in enumerate(out):
                    a = acc[k].astype(dtype)
                    a[~valid] = nodata
                    ds.write(a, 1, window=window)

        try:
            self._run_tiles(tile, windows, workers)
        finally:
            for ds in out:
                ds.close()


@njit(nogil=True, cache=True)
def _stream_graph(flat, fd, width):
    """Downstream index of each stream cell within the sorted stream cell array, or -1"""
    n = flat.shape[0]
    down = np.full(n, -1, np.int64)
    for c in range(n):
        d = fd[c]
        if d < 1 or d > 8:
            continue
        i = flat[c] // width + FD_OFF_I[d]
        j = flat[c] % width + FD_OFF_J[d]
        if i < 0 or j < 0 or j >= width:
            continue
        target = i * width + j
        lo = np.searchsorted(flat, target)
        if lo < n and flat[lo] == target:
            down[c] = lo
    return down


@njit(nogil=True, cache=True)
def _label_segments(down, keep):
    """Label stream segments between sources, junctions, and outlets

    Returns:
        tuple: Segment id for each kept cell (0 otherwise), and the number of cells
        flowing into each kept cell
    """
    n = down.shape[0]
    indeg = np.zeros(n, np.int64)
    for c in range(n):
        if keep[c] and down[c] >= 0 and keep[down[c]]:
            indeg[down[c]] += 1

    ids = np.zeros(n, np.int64)
    next_id = 1
    for c in range(n):
        # Segments begin at sources and junctions
        if not keep[c] or ids[c] != 0 or indeg[c] == 1:
            continue
        cur = c
        while True:
            ids[cur] = next_id
            nxt = down[cur]
            if nxt < 0 or not keep[nxt] or indeg[nxt] != 1:
                break
            cur = nxt
        next_id += 1

    return ids, indeg


@njit(nogil=True, cache=True)
def _prune_sources(flat, fd, down, keep, width, csx, csy, min_length):
    """Remove first-order segments shorter than a minimum length"""
    n = down.shape[0]
    indeg = np.zeros(n, np.int64)
    for c in range(n):
        if down[c] >= 0:
            indeg[down[c]] += 1

    diag = (csx**2 + csy**2) ** 0.5
    path = np.empty(n, np.int64)
    for c in range(n):
        if indeg[c] != 0:
            continue
        length = 0.0
        depth = 0
        cur = c
        while True:
            path[depth] = cur
            depth += 1
            nxt = down[cur]
            if nxt < 0:
                break
            d = fd[cur]
            if FD_OFF_I[d] != 0 and FD_OFF_J[d] != 0:
                length += diag
            elif FD_OFF_I[d] != 0:
                length += csy
            else:
                length += csx
            if indeg[nxt] != 1:
                break
            cur = nxt
        if length < min_length:
            for s in range(depth):
                keep[path[s]] = False

    return keep


def extract_streams(
    accumulation: str,
    direction: str,
    dst: str,
    threshold: float,
    min_length: float = 0,
    tile_size: int = 2048,
):
    """Extract streams where accumulation reaches a threshold, labelled with unique
    segment ids between sources, junctions, and outlets.

    Stream cells are held in memory (roughly 40 bytes per stream cell), while the
    rasters are read and written in tiles.

    Args:
        accumulation (str): Accumulation raster, for example contributing area.
        direction (str): Flow direction raster (r.watershed encoding) used to create
        the accumulation.
        dst (str): Output stream raster.
        threshold (float): Accumulation value at which streams begin, in the units of
        the accumulation raster.
        min_length (float, optional): First-order segments shorter than this length
        (raster units) are removed. Defaults to 0.
        tile_size (int, optional): Tile size in cells. Defaults to 2048.
    """
    r = Raster(direction)
    height, width = r.shape[1:]
    windows = tile_windows(height, width, tile_size)

    flat, fd = [], []
    with rasterio.open(accumulation) as acc_ds, rasterio.open(direction) as fd_ds:
        for window in windows:
            a = acc_ds.read(1, window=window, masked=True)
            cells = np.nonzero(np.ma.filled(a >= threshold, False))
            if cells[0].size == 0:
                continue
            d = fd_ds.read(1, window=window, masked=True).filled(0)[cells]
            d = np.where((d < 1) | (d > 8), 0, d)
            flat.append(
                (cells[0] + window.row_off).astype(np.int64) * width
                + cells[1]
                + window.col_off
            )
            fd.append(d.astype(np.int8))

    nodata = infer_nodata("int32")
    Raster.empty_like(dst, direction, dtype="int32", nodata=nodata)

    if not flat:
        warning("No streams were found")
        return

    flat = np.concatenate(flat)
    fd = np.concatenate(fd)
    order = np.argsort(flat)
    flat, fd = flat[order], fd[order]

    down = _stream_graph(flat, fd, width)
    keep = np.ones(flat.shape[0], np.bool_)
    if min_length > 0:
        keep = _prune_sources(
            flat, fd, down, keep, width, float(r.csx), float(r.csy), float(min_length)
        )
    ids, _ = _label_segments(down, keep)

    # Write in strips of rows, which are contiguous in the sorted stream cells
    with rasterio.open(dst, "r+") as ds:
        strip = max(1, min(height, (tile_size * tile_size) // width))
        for row0 in range(0, height, strip):
            rows = min(strip, height - row0)
            lo = np.searchsorted(flat, row0 * width)
            hi = np.searchsorted(flat, (row0 + rows) * width)
            if lo == hi:
                continue
            a = np.full((rows, width), nodata, np.int32)
            sel = ids[lo:hi] > 0
            cells = flat[lo:hi][sel] - row0 * width
            a[cells // width, cells % width] = ids[lo:hi][sel]
            ds.write(a, 1, window=Window(0, row0, width, rows))
