import dask.array as da
import unittest
from tempfile import TemporaryDirectory
import os

import dask.array as da

from tests import test_data
from hydrotools.raster import Raster, from_raster, to_raster, raster_where


class RasterTest(unittest.TestCase):
    def test_from_raster(self):
        a = from_raster(test_data.dem)

        sums = (3645919.2, 3274943.2)

        for i, result in enumerate(da.compute(da.sum(a + 1), da.sum(a - 1))):
            self.assertAlmostEqual(result, sums[i], 1)

    def test_to_raster(self):
        with TemporaryDirectory() as tmp_dir:
            a = from_raster(test_data.dem)
            out_path = os.path.join(tmp_dir, "test.tif")
            to_raster(a + 1, test_data.dem, out_path)

            self.assertTrue(os.path.isfile(out_path))

    def test_raster_where(self):
        r = Raster(test_data.dem)
        a = from_raster(r)

        a = raster_where(a < 50, 0, a)

        self.assertAlmostEqual(a.sum().compute(), 1944926.9, 1)
