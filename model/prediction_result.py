from dataclasses import dataclass
from functools import cache

import numpy as np

from core.score import ScoreName, score_func_by_name
from model.state_series import AbstractStateSeries


@dataclass(slots=True)
class PredictionResult:
    t_arr: np.ndarray  # (T,)
    u_mlt_arr: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length: T
    y_mlt_arr: np.ndarray  # (T, n_out)
    y_pred_mlt_arr: np.ndarray  # (T, n_out)
    n_washout: int

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t_arr, np.ndarray), (type(self.t_arr), self.t_arr)
        assert isinstance(self.u_mlt_arr, np.ndarray), (type(self.u_mlt_arr), self.u_mlt_arr)
        assert isinstance(self.states, AbstractStateSeries), (type(self.states), self.states)
        assert isinstance(self.y_mlt_arr, np.ndarray), (type(self.y_mlt_arr), self.y_mlt_arr)
        assert isinstance(self.y_pred_mlt_arr, np.ndarray), (type(self.y_pred_mlt_arr),
                                                             self.y_pred_mlt_arr)

        n_t = len(self.t_arr)
        assert self.t_arr.ndim == 1, self.t_arr.shape
        assert self.u_mlt_arr.ndim == 2, self.u_mlt_arr.shape
        assert self.u_mlt_arr.shape[0] == n_t, (self.u_mlt_arr.shape[0], n_t)
        assert len(self.states) == n_t, (len(self.states), n_t)
        assert self.y_mlt_arr.ndim == 2, self.y_mlt_arr.shape
        assert self.y_mlt_arr.shape[0] == n_t, (self.y_mlt_arr.shape[0], n_t)
        assert self.y_pred_mlt_arr.ndim == 2, self.y_pred_mlt_arr.shape
        assert self.y_pred_mlt_arr.shape[0] == n_t, (self.y_pred_mlt_arr.shape[0], n_t)

        # copy arrays and make readonly
        self.t_arr = self.t_arr.copy()
        self.t_arr.setflags(write=False)
        self.u_mlt_arr = self.u_mlt_arr.copy()
        self.u_mlt_arr.setflags(write=False)
        self.y_mlt_arr = self.y_mlt_arr.copy()
        self.y_mlt_arr.setflags(write=False)
        self.y_pred_mlt_arr = self.y_pred_mlt_arr.copy()
        self.y_pred_mlt_arr.setflags(write=False)

    def washout_masked(self) -> "PredictionResult":
        s = slice(self.n_washout, None)
        return PredictionResult(
            t_arr=self.t_arr[s],
            u_mlt_arr=self.u_mlt_arr[s],
            states=self.states[s],
            y_mlt_arr=self.y_mlt_arr[s],
            y_pred_mlt_arr=self.y_pred_mlt_arr[s],
            n_washout=0,
        )

    @cache
    def score(
            self,
            score: ScoreName,
    ) -> np.ndarray:
        """
        このPredictionResultインスタンスの予測結果に対するスコアを計算する。

        Args:
            score (Literal["r2", "capacity", "mse"]): 計算するスコアの種類

        Returns:
            np.ndarray: スコア値（shape: (n_out,)）
        """
        score_func = score_func_by_name(score, "2d")
        return score_func(y_true=self.y_mlt_arr, y_pred=self.y_pred_mlt_arr)


class PredictionResultSet:
    def __init__(self, results: list[PredictionResult]):
        self._results = results

    def __len__(self):
        return len(self._results)

    def __getitem__(self, sample_index: int):
        return self._results[sample_index]

    def washout_masked(self) -> "PredictionResultSet":
        return PredictionResultSet([r.washout_masked() for r in self._results])

    # TODO: aggregated_score must return float
    @cache
    def aggregated_score(
            self,
            score: ScoreName,
    ) -> np.ndarray:
        """
        結果全体のデータをひとつのデータとして集約し、scoreを計算する。
        """
        y_true_arr = np.array([r.y_mlt_arr for r in self._results])
        y_pred_arr = np.array([r.y_pred_mlt_arr for r in self._results])
        score_func = score_func_by_name(score)
        return score_func(y_true=y_true_arr, y_pred=y_pred_arr)
