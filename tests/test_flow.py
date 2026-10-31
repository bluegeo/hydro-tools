import os
import unittest
from tempfile import TemporaryDirectory

from tests import test_data
from numba import njit
import numpy as np
import rasterio

from hydrotools.flow import (
    FD_OFF_I,
    FD_OFF_J,
    TiledAccumulation,
    _perimeter_cells,
    _perimeter_index,
    extract_streams,
)
from hydrotools.watershed import flow_direction


@njit
def reference_accumulation(fd, weights):
    """Accumulate by walking every cell to the end of its flow path"""
    acc = weights.astype(np.float64).copy()
    h, w = fd.shape
    for i in range(h):
        for j in range(w):
            ci, cj = i, j
            while True:
                d = fd[ci, cj]
                if d < 1 or d > 8:
                    break
                ci, cj = ci + FD_OFF_I[d], cj + FD_OFF_J[d]
                if not (0 <= ci < h and 0 <= cj < w):
                    break
                acc[ci, cj] += weights[i, j]
    return acc


class FlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = TemporaryDirectory()
        cls.direction = os.path.join(cls.tmp_dir.name, "fd.tif")
        flow_direction(test_data.dem, cls.direction)

        with rasterio.open(cls.direction) as ds:
            a = ds.read(1, masked=True)
            cls.valid = ~np.ma.getmaskarray(a)
            fd = a.filled(0)
            cls.fd = np.where((fd < 1) | (fd > 8), 0, fd)
            cls.profile = ds.profile

        rng = np.random.default_rng(1)
        cls.weights = np.where(cls.valid, rng.integers(1, 10, cls.fd.shape), 0)
        cls.weight_raster = os.path.join(cls.tmp_dir.name, "weights.tif")
        profile = dict(cls.profile, dtype="float64", nodata=None)
        with rasterio.open(cls.weight_raster, "w", **profile) as ds:
            ds.write(cls.weights.astype(np.float64), 1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def read(self, path):
        with rasterio.open(path) as ds:
            return ds.read(1, masked=True)

    def test_perimeter_ordering(self):
        for h, w in ((1, 1), (1, 5), (5, 1), (2, 2), (3, 4), (7, 7)):
            rows, cols = _perimeter_cells(h, w)
            self.assertEqual(len(set(zip(rows, cols))), rows.shape[0])
            for p, (i, j) in enumerate(zip(rows, cols)):
                self.assertEqual(_perimeter_index(h, w, i, j), p)
            interior = [
                (i, j)
                for i in range(h)
                for j in range(w)
                if (i, j) not in set(zip(rows, cols))
            ]
            for i, j in interior:
                self.assertEqual(_perimeter_index(h, w, i, j), -1)

    def test_matches_reference(self):
        expected = reference_accumulation(self.fd, self.weights)
        for tile_size in (50, 300, 4096):
            dst = os.path.join(self.tmp_dir.name, f"acc_{tile_size}.tif")
            TiledAccumulation(self.direction, tile_size=tile_size).accumulate(
                [self.weight_raster], [dst], dtype="float64"
            )
            a = self.read(dst)
            self.assertTrue((np.ma.getmaskarray(a) == ~self.valid).all())
            np.testing.assert_array_equal(a.filled(0)[self.valid], expected[self.valid])

    def test_seeds_and_bands(self):
        seeds = [(10, 20, [100.0, 1.0]), (200, 300, [5000.0, 0.0]), (0, 0, [7.0, 7.0])]
        seeded = self.weights.astype(np.float64)
        for row, col, values in seeds:
            seeded[row, col] += values[0]
        expected = reference_accumulation(self.fd, seeded)
        expected_count = reference_accumulation(self.fd, self.valid.astype(np.float64))
        for row, col, values in seeds:
            expected_count += reference_accumulation(
                self.fd, np.eye(1, self.fd.size, row * self.fd.shape[1] + col).reshape(self.fd.shape) * values[1]
            )

        acc = TiledAccumulation(self.direction, tile_size=64, max_memory=64)
        dst = [os.path.join(self.tmp_dir.name, f) for f in ("seeded.tif", "count.tif")]
        acc.accumulate([self.weight_raster, 1.0], dst, seeds=seeds, dtype="float64")
        np.testing.assert_array_equal(self.read(dst[0]).filled(0)[self.valid], expected[self.valid])
        np.testing.assert_array_equal(
            self.read(dst[1]).filled(0)[self.valid], expected_count[self.valid]
        )

        # The first pass is reused when only the seeds change
        pass1 = acc._pass1
        acc.accumulate([self.weight_raster, 1.0], dst, dtype="float64")
        self.assertIs(acc._pass1, pass1)
        np.testing.assert_array_equal(
            self.read(dst[0]).filled(0)[self.valid],
            reference_accumulation(self.fd, self.weights)[self.valid],
        )

    def test_extract_streams(self):
        count = os.path.join(self.tmp_dir.name, "cells.tif")
        TiledAccumulation(self.direction, tile_size=128).accumulate(
            [1.0], [count], dtype="float64"
        )
        c = self.read(count)

        streams = os.path.join(self.tmp_dir.name, "streams.tif")
        extract_streams(count, self.direction, streams, 1000, tile_size=100)
        s = self.read(streams)
        np.testing.assert_array_equal(~np.ma.getmaskarray(s), c.filled(0) >= 1000)

        # Segment ids are unique and contiguous
        ids = np.unique(s.compressed())
        self.assertTrue((ids == np.arange(1, ids.size + 1)).all())
        self.assertGreater(ids.size, 1)

        # Pruning removes short first-order segments only
        pruned = os.path.join(self.tmp_dir.name, "pruned.tif")
        extract_streams(count, self.direction, pruned, 1000, min_length=250, tile_size=100)
        p = self.read(pruned)
        self.assertLess(p.count(), s.count())
        self.assertTrue((~np.ma.getmaskarray(p) <= ~np.ma.getmaskarray(s)).all())
