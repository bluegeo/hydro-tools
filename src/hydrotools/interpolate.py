from multiprocessing.dummy import Pool as DummyPool
from multiprocessing import cpu_count

import numpy as np
from numba import jit
from sklearn.neighbors import KNeighborsRegressor, BallTree

from hydrotools.utils import GrassRunner


def fill_nodata(raster_source: str, destination: str, method: str = "idw"):
    """Interpolate regions with no data in a raster

    Args:
        raster_source (str): [description]
        destination (str): Destination raster dataset
        method (str, optional): [description]. Defaults to "linear".
    """
    if method == "idw":
        with GrassRunner(raster_source) as gr:
            gr.run_command(
                "r.fill.stats",
                (raster_source, "input", "raster"),
                input="input",
                output="output",
            )
            gr.save_raster("output", destination)


def cubic_spline(
    point_source: str,
    template_raster: str,
    destination_raster: str,
    column: str = None,
    method: str = "bicubic",
):
    """Perform interpolation of a set of vector points into a raster grid
    using a b-spline.

    Args:
        point_source (str): Vector point dataset
        template_raster (str): Raster to collect spatial parameters from
        destination_raster (str): Output raster dataset
        column (str): Column name to use as z-values. Defaults to None, whereby the
        geometry z-values will be used.
        method (str, optional): Spline method. Choose from "bilinear" and "bicubic".
        Defaults to "bicubic".
    """
    with GrassRunner(template_raster) as gr:
        gr.run_command(
            "v.surf.bspline",
            (point_source, "points", "vector"),
            input="points",
            column=column,
            method=method,
            raster_output="interpolated",
        )
        gr.save_raster("interpolated", destination_raster)


class PointInterpolator:
    """Utility for custom interpolation"""

    def __init__(self, obs: np.ndarray, obs_z: np.ndarray, pred: np.ndarray):
        self.obs = np.asarray(obs, dtype="float64")
        self.obs_z = np.asarray(obs_z, dtype="float64")
        self.pred = np.asarray(pred, dtype="float64")

        if obs.shape[0] != obs_z.shape[0]:
            raise ValueError(
                "Number of observed points and observed values (z) do not match"
            )

        if obs.size == 0:
            raise ValueError("At least one observed point required")

    def chunks(self, n_chunks: int, return_obs: bool, *args):
        """Generator for chunking and iterating data points to predict

        Args:
            n_chunks (int): Number of chunks
            return_obs (bool): Include observed data in returned generator

        Yields:
            Generator[np.ndarray]: Chunk of predicted points (with obs and obs_z if required)
        """
        if n_chunks < 1:
            n_chunks = 1
        chunkRange = list(range(0, self.pred.shape[0] + n_chunks, n_chunks))
        for fr, to in zip(chunkRange[:-1], chunkRange[1:-1] + [self.pred.shape[0]]):
            if return_obs:
                yield (self.obs, self.obs_z, np.atleast_2d(self.pred[fr:to])) + args
            else:
                yield np.atleast_2d(self.pred[fr:to])

    def map_chunks(
        self,
        func,
        *args,
        n_chunks: int = 500,
        return_obs: bool = False,
        run_sync: bool = False,
    ) -> np.ndarray:
        """Map chunks of predicted points to an interpolation function

        Args:
            func (callable): Interpolator
            n_chunks (int, optional): Number of chunks to use. Defaults to 500.
            return_obs (bool, optional): Return obs and obs_z for method. Defaults to False.
            run_sync (bool, optional): Run synchronously. Defaults to False.

        Returns:
            np.ndarray: Predicted values
        """
        if run_sync:
            values = [func(a) for a in self.chunks(n_chunks, return_obs, *args)]
        else:
            p = DummyPool(cpu_count())
            values = list(p.imap(func, self.chunks(n_chunks, return_obs, *args)))

        return np.concatenate(values)

    def full_idw(self) -> np.ndarray:
        """Perform idw interpolation using all observed data for every predicted point"""
        return self.map_chunks(full_idw, return_obs=True)

    def idw(self, n_neighbours=3) -> np.ndarray:
        """IDW interpolation"""
        n_neighbours = min(n_neighbours, self.obs_z.size)

        def idw_callable(distance):
            return 1 / (distance ** 2)

        knn = KNeighborsRegressor(n_neighbors=n_neighbours, weights=idw_callable).fit(
            self.obs, self.obs_z
        )

        return self.map_chunks(knn.predict)

    def bounding_idw(self, n_neighbours=4) -> np.ndarray:
        """Perform IDW interpolation, dynamically selecting a number of neighbours to
        ensure each predicted point is within the bounding box of the neighbours.

        Returns:
            np.ndarray: Array of predicted values
        """
        return bidw_task(self.obs, self.obs_z, self.pred, n_neighbours)

    def mean(self, n_neighbours=3) -> np.ndarray:
        """Perform interpolation using a mean of a number of surrounding points

        Args:
            n_neighbours (int, optional): Number of closest neighbours to use.
            Defaults to 3.

        Returns:
            [np.ndarray]: Array of predicted values
        """
        n_neighbours = min(n_neighbours, self.obs_z.size)

        knn = KNeighborsRegressor(n_neighbors=n_neighbours).fit(self.obs, self.obs_z)

        return self.map_chunks(knn.predict)


def bidw_task(obs, obs_z, pred, n_neighbours):
    obs = obs.copy()
    obs_z = obs_z.copy()
    pred_z = np.full(pred.shape[0], np.nan, "float32")

    def bounds(pred_pnts, obs_pnts):
        top = obs_pnts[..., 1].max(1)
        bottom = obs_pnts[..., 1].min(1)
        left = obs_pnts[..., 0].min(1)
        right = obs_pnts[..., 0].max(1)
        return (
            (pred_pnts[:, 0] <= right)
            & (pred_pnts[:, 0] >= left)
            & (pred_pnts[:, 1] >= bottom)
            & (pred_pnts[:, 1] <= top)
        )

    def idw(obs_values, distances):
        idw_result = np.full(obs_values.shape[0], np.nan, "float32")

        zero_dist = np.any(distances == 0, 1)
        zero_dist_locs = np.argmin(distances[zero_dist], 1)
        idw_result[zero_dist] = obs_values[zero_dist][
            (np.arange(zero_dist_locs.shape[0]), zero_dist_locs)
        ]

        weights = 1 / distances[~zero_dist] ** 2
        idw_result[~zero_dist] = np.sum(
            obs_values[~zero_dist] * (weights / weights.sum(1, keepdims=True)), axis=1
        )

        return idw_result

    pred_remaining = np.isnan(pred_z)
    completeness_tracking = []
    while True:
        bt = BallTree(obs)

        distances, indices = bt.query(pred[pred_remaining], n_neighbours)

        bounded = bounds(pred[pred_remaining], obs[indices])

        locs = np.where(pred_remaining)
        locs = locs[0][bounded]
        pred_z[locs] = idw(obs_z[indices][bounded], distances[bounded])

        pred_remaining = np.isnan(pred_z)
        remaining_points = pred_remaining.sum()

        obs = np.concatenate([obs, pred[locs]])
        obs_z = np.concatenate([obs_z, pred_z[locs]])

        completeness = round(
            ((pred_z.shape[0] - float(remaining_points)) / pred_z.shape[0]) * 100, 2
        )
        completeness_tracking.append(completeness)
        print(f"{completeness}%")
        if len(completeness_tracking) > 2 and len(set(completeness_tracking[-3:])) == 1:
            break

    if remaining_points > 0:
        knn = KNeighborsRegressor(n_neighbors=n_neighbours).fit(obs, obs_z)
        pred_z[pred_remaining] = knn.predict(pred[pred_remaining])

    return pred_z


@jit(nopython=True, nogil=True)
def full_idw(args):
    """Complete outer product idw

    Args:
        args (tuple): (obs, obs_z, pred)
    """
    obs, obs_z, pred = args

    obs_rows = obs.shape[0]
    weights = np.zeros(obs_rows, np.float64)

    pred_rows = pred.shape[0]
    output = np.zeros(pred_rows, np.float64)

    for pred_row in range(pred_rows):
        pred_x = pred[pred_row, 0]
        pred_y = pred[pred_row, 1]

        weights_sum = 0
        coincident = np.nan
        for obs_row in range(obs_rows):
            _x = obs[obs_row, 0]
            _y = obs[obs_row, 1]

            distance = np.sqrt((_x - pred_x) ** 2 + (_y - pred_y) ** 2)

            if distance == 0:
                coincident = obs_z[obs_row]
                break
            else:
                weight = 1 / distance ** 2

            weights[obs_row] = weight
            weights_sum += weight

        if not np.isnan(coincident):
            output[pred_row] = coincident
        else:
            output[pred_row] = np.sum(obs_z * weights / weights_sum)

    return output
