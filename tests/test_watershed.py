import os
import unittest
from tempfile import TemporaryDirectory

from tests import test_data
from hydrotools.watershed import flow_direction_accumulation, auto_basin


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
                1E5,
                os.path.join(tmp_dir, "basins.tif"),
            )
