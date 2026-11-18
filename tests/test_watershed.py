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

    def test_expanded_flow_accumulation(self):
        from hydrotools.watershed import ExpandedFlowAccumulation, FlowAccumulation

        with TemporaryDirectory() as tmp_dir:
            efa = ExpandedFlowAccumulation(test_data.dem, snap_distance=100.0, tile_size=256)
            self.assertGreater(len(efa.inflows), 0)
            self.assertTrue(all(cell is not None for cell in efa.cells))

            streams = os.path.join(tmp_dir, "streams.tif")
            direction = os.path.join(tmp_dir, "direction.tif")
            ca = os.path.join(tmp_dir, "ca.tif")
            efa.extract_streams(streams, direction, min_area=1e5)
            efa.contributing_area(ca)
            efa.calculate(test_data.precip, os.path.join(tmp_dir, "precip.tif"), "mean")
            efa.save_inflows(os.path.join(tmp_dir, "inflows.gpkg"))

            # The inflow area is added to the contributing area at the outlet
            FlowAccumulation(direction).contributing_area(os.path.join(tmp_dir, "plain.tif"))
            with rasterio.open(ca) as a, rasterio.open(os.path.join(tmp_dir, "plain.tif")) as b:
                added = (a.read(1, masked=True) - b.read(1, masked=True)).filled(0)
            self.assertAlmostEqual(added.max(), efa.inflows.area.sum() / 1e6, places=6)

            with rasterio.open(streams) as s:
                self.assertGreater(s.read(1, masked=True).count(), 0)
