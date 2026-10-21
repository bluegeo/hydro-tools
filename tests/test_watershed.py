import os
import unittest
from tempfile import TemporaryDirectory

from tests import test_data
import numpy as np
import rasterio

from hydrotools.watershed import (
    flow_direction_accumulation,
    auto_basin,
    breach_depressions,
)


class WSTest(unittest.TestCase):
    def test_fa_fd(self):
        with TemporaryDirectory() as tmp_dir:
            flow_direction_accumulation(
                test_data.dem,
                os.path.join(tmp_dir, "fd.tif"),
                os.path.join(tmp_dir, "fa.tif"),
            )

    def test_auto_basin(self):
        with TemporaryDirectory() as tmp_dir:
            auto_basin(
                test_data.dem,
                1e5,
                os.path.join(tmp_dir, "basins.tif"),
            )

    def test_breach_depressions(self):
        try:
            import whitebox  # noqa
        except ImportError:
            self.skipTest("whitebox is not installed")

        with TemporaryDirectory() as tmp_dir:
            single = os.path.join(tmp_dir, "single.tif")
            chunked = os.path.join(tmp_dir, "chunked.tif")
            breach_depressions(test_data.dem, single, 100, fill=False)
            # Force chunking with a small amount of memory
            breach_depressions(test_data.dem, chunked, 100, max_memory=2, fill=False)

            with rasterio.open(single) as a, rasterio.open(chunked) as b:
                self.assertEqual(a.profile["compress"], "lzw")
                self.assertEqual(a.shape, b.shape)
                diff = np.abs(a.read(1) - b.read(1)) > 1e-3
                self.assertLess(diff.mean(), 0.001)
