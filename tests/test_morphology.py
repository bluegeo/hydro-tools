import os
import unittest
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

from hydrotools.morphology import (
    DEFAULT_COST_FLOOR,
    RiparianConnectivity,
    convergence_weight,
    region_width,
    riparian_zones,
    rise_cost_surface,
    slope_modifier,
)
from hydrotools.raster import from_raster
from tests import test_data

CELL_SIZE = 5.0


def write_raster(path, data, nodata):
    data = np.asarray(data)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype.name,
        crs="EPSG:32610",
        transform=from_origin(500000.0, 5000000.0, CELL_SIZE, CELL_SIZE),
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)
    return path


def load(path):
    return np.ma.squeeze(from_raster(path).compute())


def valley_dem(rows=200, cols=400, lateral=0.01, longitudinal=0.005, hollow=None):
    """A planar V-shaped valley draining east along the centre row.

    `hollow=(col, half_width, gradient)` carves a tributary hollow into the north side
    whose axis climbs at `gradient` (gentler than `lateral`) from the valley floor.
    """
    i, j = np.indices((rows, cols), dtype="float64")
    axis = rows // 2
    z = 100.0 - longitudinal * j * CELL_SIZE + lateral * np.abs(i - axis) * CELL_SIZE

    if hollow is not None:
        col, half_width, gradient = hollow
        z_floor = 100.0 - longitudinal * col * CELL_SIZE
        z_hollow = (
            z_floor
            + gradient * (axis - i) * CELL_SIZE
            + lateral * np.abs(j - col) * CELL_SIZE
        )
        north = (i < axis) & (np.abs(j - col) <= half_width)
        z = np.where(north, np.minimum(z, z_hollow), z)

    return z.astype("float32")


class SlopeModifierTest(unittest.TestCase):
    def test_bounds_and_monotonic(self):
        gradient = da.from_array(np.array([0.0, 0.015, 0.0275, 0.04, 0.2]))
        g = slope_modifier(gradient, (0.015, 0.04), (0.5, 1.25)).compute()

        np.testing.assert_allclose(g, [1.25, 1.25, 0.875, 0.5, 0.5])
        self.assertTrue(np.all(np.diff(g) <= 0))

    def test_invalid_arguments(self):
        gradient = da.from_array(np.array([0.01]))
        with self.assertRaises(ValueError):
            slope_modifier(gradient, (0.04, 0.015))
        with self.assertRaises(ValueError):
            slope_modifier(gradient, g_range=(1.25, 0.5))


class RiparianZonesTest(unittest.TestCase):
    def test_classification_from_seeds(self):
        # Seed A has c0 = 0 (budget T3 = 4), seed B has c0 = 1 (budget T3 = 3).
        # With h3 = 1, f1 = 1, f2 = 3: A -> T1 = 1, T2 = 3; B -> T1 = 2/3, T2 = 2.
        nodata = np.finfo("float32").min
        nearest = np.array([[0.0] * 6, [1.0] * 6], "float32")
        rise = np.array(
            [[0.0, 0.99, 1.01, 2.99, 3.01, 3.99], [0.0, 0.6, 0.7, 1.99, 2.01, 2.99]],
            "float32",
        )
        cost = nearest + rise
        cost[1, 5] = nodata

        with TemporaryDirectory() as tmp_dir:
            cost_src = write_raster(os.path.join(tmp_dir, "cost.tif"), cost, nodata)
            nearest_src = write_raster(
                os.path.join(tmp_dir, "nearest.tif"), nearest, nodata
            )
            zones_dst = os.path.join(tmp_dir, "zones.tif")
            score_dst = os.path.join(tmp_dir, "score.tif")

            riparian_zones(cost_src, nearest_src, 4.0, 1.0, 3.0, 1.0, zones_dst, score_dst)

            zones = load(zones_dst)
            score = load(score_dst)

        expected = np.ma.masked_equal(
            np.array([[1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 0]]), 0
        )
        np.testing.assert_array_equal(np.ma.filled(zones, 0), np.ma.filled(expected, 0))
        np.testing.assert_array_equal(
            np.ma.getmaskarray(zones), np.ma.getmaskarray(expected)
        )

        self.assertAlmostEqual(float(score[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(score[0, 5]), 1 - 3.99 / 4.0, places=5)
        self.assertAlmostEqual(float(score[1, 4]), 1 - 2.01 / 3.0, places=5)
        self.assertTrue(np.ma.is_masked(score[1, 5]))


class RiseCostSurfaceTest(unittest.TestCase):
    def test_flat_extent_bounded_by_cost_floor(self):
        rows, cols, stream_col = 60, 200, 100
        budget = 0.5
        nodata = np.finfo("float32").min

        dem = np.zeros((rows, cols), "float32")
        start_cost = np.full((rows, cols), nodata, "float32")
        start_cost[:, stream_col] = 0

        with TemporaryDirectory() as tmp_dir:
            dem_src = write_raster(os.path.join(tmp_dir, "dem.tif"), dem, nodata)
            start_src = write_raster(
                os.path.join(tmp_dir, "start.tif"), start_cost, nodata
            )
            cost_dst = os.path.join(tmp_dir, "cost.tif")
            nearest_dst = os.path.join(tmp_dir, "nearest.tif")
            zones_dst = os.path.join(tmp_dir, "zones.tif")

            rise_cost_surface(dem_src, start_src, cost_dst, nearest_dst, memory=512)
            riparian_zones(cost_dst, nearest_dst, budget, 1.0, 3.0, 0.4, zones_dst)

            cost = load(cost_dst)
            zones = load(zones_dst)

        per_cell = DEFAULT_COST_FLOOR * CELL_SIZE
        self.assertAlmostEqual(float(cost[30, stream_col + 10]), 10 * per_cell, places=4)

        # Cells reached with cost strictly below the budget are in the outer zone
        reach = int(np.ceil(budget / per_cell)) - 1
        in_zone = ~np.ma.getmaskarray(zones[30])
        self.assertEqual(in_zone.sum(), 2 * reach + 1)
        self.assertTrue(np.all(np.ma.getmaskarray(zones[30, : stream_col - reach])))


class RiparianConnectivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = TemporaryDirectory()
        nodata = -9999.0
        rows, cols = 200, 400

        cls.dem = write_raster(
            os.path.join(cls.tmp_dir.name, "dem.tif"), valley_dem(rows, cols), nodata
        )
        cls.precip = write_raster(
            os.path.join(cls.tmp_dir.name, "precip.tif"),
            np.full((rows, cols), 3000.0, "float32"),
            nodata,
        )

        cls.rc = RiparianConnectivity(cls.dem, cls.precip)
        cls.rc.extract_streams(min_ws_area=1e5, memory=1024)
        cls.rc.calc_stream_attributes()
        cls.rc.calc_zones(memory=1024)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_requires_attributes_before_zones(self):
        rc = RiparianConnectivity(self.dem, self.precip)
        with self.assertRaises(AttributeError):
            rc.calc_stream_attributes()
        with self.assertRaises(AttributeError):
            rc.calc_zones()
        with self.assertRaises(AttributeError):
            rc.calc_connectivity(os.path.join(self.tmp_dir.name, "none.gpkg"))

    def test_streams_are_zone_one(self):
        zones = load(self.rc.zones)
        score = load(self.rc.score)
        on_stream = ~np.ma.getmaskarray(load(self.rc.streams))

        self.assertTrue(np.all(zones[on_stream] == 1))
        self.assertTrue(np.all(score[on_stream] == 1))

    def test_zones_are_nested(self):
        zones = np.ma.filled(load(self.rc.zones), 0)
        structure = np.ones((3, 3), bool)

        around_one = ndimage.binary_dilation(zones == 1, structure)
        self.assertTrue(np.all(np.isin(zones[around_one], [1, 2])))

        around_two = ndimage.binary_dilation(zones == 2, structure)
        self.assertTrue(np.all(np.isin(zones[around_two], [1, 2, 3])))

        region = load(self.rc.region)
        np.testing.assert_array_equal(~np.ma.getmaskarray(region), zones > 0)

    def test_floodplain_widens_downstream_and_fringe_does_not(self):
        zones = np.ma.filled(load(self.rc.zones), 0)

        floodplain = (zones > 0) & (zones < 3)
        fringe = zones == 3

        upstream = slice(100, 150)
        downstream = slice(300, 350)

        fp_up = floodplain[:, upstream].sum(axis=0).mean()
        fp_down = floodplain[:, downstream].sum(axis=0).mean()
        self.assertGreater(fp_down, fp_up * 1.2)

        fr_up = fringe[:, upstream].sum(axis=0).mean()
        fr_down = fringe[:, downstream].sum(axis=0).mean()
        self.assertAlmostEqual(fr_down / fr_up, 1.0, delta=0.15)

    def test_gradient_modifier_scales_floodplain(self):
        zones = np.ma.filled(load(self.rc.zones), 0)
        baseline = ((zones > 0) & (zones < 3)).sum()

        rc = RiparianConnectivity(self.dem, self.precip)
        rc.streams = self.rc.streams
        rc.flow_direction = self.rc.flow_direction
        rc.flow_accumulation = self.rc.flow_accumulation
        rc.bankfull_width = self.rc.bankfull_width
        rc.bankfull_depth = self.rc.bankfull_depth
        rc.stream_slope = self.rc.stream_slope

        rc.calc_zones(g_range=(0.5, 0.5), memory=1024)
        narrow = np.ma.filled(load(rc.zones), 0)
        narrow = ((narrow > 0) & (narrow < 3)).sum()

        self.assertLess(narrow, baseline)

    def test_region_width_matches_cross_sections(self):
        zones = np.ma.filled(load(self.rc.zones), 0)
        floodplain = (zones > 0) & (zones < 3)

        width_dst = os.path.join(self.tmp_dir.name, "width.tif")
        region_width(
            da.from_array(floodplain.reshape((1,) + floodplain.shape)),
            self.rc.streams,
            width_dst,
        )
        width = load(width_dst)
        on_stream = ~np.ma.getmaskarray(load(self.rc.streams))
        self.assertTrue(np.all(~np.ma.getmaskarray(width)[on_stream]))

        # Compare with the floodplain column counts along the straight valley axis
        for cols in (slice(100, 150), slice(300, 350)):
            measured = np.asarray(width[:, cols][on_stream[:, cols]]).mean()
            expected = floodplain[:, cols].sum(axis=0).mean() * CELL_SIZE
            self.assertAlmostEqual(measured / expected, 1.0, delta=0.15)

    def test_outputs(self):
        vector_dst = os.path.join(self.tmp_dir.name, "zones.gpkg")
        raster_dst = os.path.join(self.tmp_dir.name, "zones_out.tif")
        score_dst = os.path.join(self.tmp_dir.name, "score_out.tif")

        self.rc.calc_connectivity(vector_dst, score_dst=score_dst)
        self.rc.calc_connectivity(raster_dst, vector=False)

        self.assertTrue(os.path.exists(vector_dst))
        np.testing.assert_array_equal(
            np.ma.filled(load(raster_dst), 0), np.ma.filled(load(self.rc.zones), 0)
        )
        score = load(score_dst)
        self.assertGreaterEqual(float(score.min()), 0)
        self.assertLessEqual(float(score.max()), 1)


class ConvergenceFringeTest(unittest.TestCase):
    hollow_col = 300
    hollow_half_width = 10
    hollow_gradient = 0.015
    lateral = 0.03

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = TemporaryDirectory()
        nodata = -9999.0
        rows, cols = 200, 400

        dem = write_raster(
            os.path.join(cls.tmp_dir.name, "dem.tif"),
            valley_dem(
                rows,
                cols,
                lateral=cls.lateral,
                hollow=(cls.hollow_col, cls.hollow_half_width, cls.hollow_gradient),
            ),
            nodata,
        )
        precip = write_raster(
            os.path.join(cls.tmp_dir.name, "precip.tif"),
            np.full((rows, cols), 3000.0, "float32"),
            nodata,
        )

        cls.rc = RiparianConnectivity(dem, precip)
        # The hollow drains less than the initiation area, so it is not a stream
        cls.rc.extract_streams(min_ws_area=2e5, memory=1024)
        cls.rc.calc_stream_attributes()

        cls.rc.calc_zones(memory=1024)
        cls.rise_zones = np.ma.filled(load(cls.rc.zones), 0)
        cls.rise_score = np.ma.filled(load(cls.rc.score), 0)

        cls.rc.calc_zones(fringe_mode="convergence", a_hill=5e3, memory=1024)
        cls.conv_zones = np.ma.filled(load(cls.rc.zones), 0)
        cls.conv_score = np.ma.filled(load(cls.rc.score), 0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_hollow_is_not_a_stream(self):
        streams = ~np.ma.getmaskarray(load(self.rc.streams))
        north = slice(0, 100)
        hollow = slice(
            self.hollow_col - self.hollow_half_width,
            self.hollow_col + self.hollow_half_width + 1,
        )
        self.assertFalse(streams[north, hollow].any())

    def test_floodplain_unchanged_and_fringe_is_superset(self):
        floodplain = (self.rise_zones > 0) & (self.rise_zones < 3)
        np.testing.assert_array_equal(
            self.rise_zones[floodplain], self.conv_zones[floodplain]
        )
        self.assertTrue(np.all((self.rise_zones > 0) <= (self.conv_zones > 0)))
        self.assertTrue(np.all(self.conv_score >= self.rise_score - 1e-6))

    def test_fringe_extends_in_hollow_only(self):
        north = slice(0, 100)
        hollow = slice(
            self.hollow_col - self.hollow_half_width,
            self.hollow_col + self.hollow_half_width + 1,
        )
        planar = slice(100, 200)

        rise_hollow = (self.rise_zones[north, hollow] == 3).sum()
        conv_hollow = (self.conv_zones[north, hollow] == 3).sum()
        self.assertGreater(conv_hollow, rise_hollow)

        np.testing.assert_array_equal(
            self.rise_zones[north, planar], self.conv_zones[north, planar]
        )

        rise_reach = np.where(self.rise_zones[north, self.hollow_col] == 3)[0].min()
        conv_reach = np.where(self.conv_zones[north, self.hollow_col] == 3)[0].min()
        self.assertLess(conv_reach, rise_reach)

    def test_fringe_outputs_recorded(self):
        self.assertIsNotNone(self.rc.fringe_scale)
        scale = load(self.rc.fringe_scale)
        self.assertGreaterEqual(float(scale.min()), 1.0)
        self.assertLessEqual(float(scale.max()), 4.0 + 1e-6)

    def test_fringe_is_connected_to_floodplain(self):
        region = self.conv_zones > 0
        labels, n = ndimage.label(region, np.ones((3, 3)))
        anchored = np.unique(labels[(self.conv_zones > 0) & (self.conv_zones < 3)])
        self.assertEqual(set(np.unique(labels[region])), set(anchored))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.rc.calc_zones(fringe_mode="buffer")


class ConvergenceWeightTest(unittest.TestCase):
    def test_weight_bounds(self):
        fa = da.from_array(np.array([0.0, 10.0, 40.0, 160.0, 1e6]))
        w = convergence_weight(fa, 25.0, 1000.0, gamma=0.5, w_min=0.25).compute()

        # 40 cells * 25 m2 = a_hill -> 1; 160 cells -> (1/4) ** 0.5
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0, 0.5, 0.25])

    def test_invalid_arguments(self):
        fa = da.from_array(np.array([1.0]))
        with self.assertRaises(ValueError):
            convergence_weight(fa, 25.0, 0.0)
        with self.assertRaises(ValueError):
            convergence_weight(fa, 25.0, 100.0, w_min=0.0)


class EngineComparisonTest(unittest.TestCase):
    """A levee parallel to the stream costs its height once with `walk` and twice
    with `cost`, so the walk engine reaches further beyond it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = TemporaryDirectory()
        nodata = -9999.0
        rows, cols = 200, 400
        axis = rows // 2

        dem = valley_dem(rows, cols)
        for offset in (-22, 22):
            dem[axis + offset - 2 : axis + offset + 3, :] += 0.8

        cls.dem = write_raster(os.path.join(cls.tmp_dir.name, "dem.tif"), dem, nodata)
        cls.precip = write_raster(
            os.path.join(cls.tmp_dir.name, "precip.tif"),
            np.full((rows, cols), 3000.0, "float32"),
            nodata,
        )

        cls.rc = RiparianConnectivity(cls.dem, cls.precip)
        cls.rc.extract_streams(min_ws_area=1e5, memory=1024)
        cls.rc.calc_stream_attributes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def extent(self, **kwargs):
        self.rc.calc_zones(memory=1024, **kwargs)
        region = ~np.ma.getmaskarray(load(self.rc.zones))
        return region[:, 200:350].sum(axis=0).mean()

    def test_walk_reaches_further_than_cost(self):
        walk = self.extent(engine="walk")
        cost = self.extent(engine="cost")
        self.assertGreater(walk, cost * 1.1)

    def test_smoothing_runs(self):
        smoothed = self.extent(engine="walk", smooth_dist=15.0)
        self.assertGreater(smoothed, 0)
        self.assertIsNotNone(self.rc.cost_dem)

    def test_invalid_engine(self):
        with self.assertRaises(ValueError):
            self.rc.calc_zones(engine="hike")


class RiparianConnectivityDataTest(unittest.TestCase):
    def test_workflow_on_sample_data(self):
        with TemporaryDirectory() as tmp_dir:
            rc = RiparianConnectivity(test_data.dem, test_data.precip)
            rc.extract_streams(min_ws_area=1e5, memory=1024)
            rc.calc_stream_attributes()
            rc.calc_zones(memory=1024)

            zones = load(rc.zones)
            self.assertEqual(set(np.unique(np.ma.compressed(zones))), {1, 2, 3})

            dst = os.path.join(tmp_dir, "zones.gpkg")
            rc.calc_connectivity(dst)
            self.assertTrue(os.path.exists(dst))
