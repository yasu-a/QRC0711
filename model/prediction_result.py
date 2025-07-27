from functools import cache

import numpy as np

from core.score import ScoreName, score_func_by_name
from model.state_array import AbstractState2DArray


class PredictionResult:
    def __init__(
            self,
            *,
            t_seq: np.ndarray,  # (T,)
            u_seq_n: np.ndarray,  # (T, n_in)
            x_seq_n: AbstractState2DArray,  # length: (T, n_state)
            y_true_seq_n: np.ndarray,  # (T, n_out)
            y_pred_seq_n: np.ndarray,  # (T, n_out)
            n_washout: int,
            readonly_without_copy=False,
    ):
        """
        予測結果を保持するクラス。

        Args:
            t_seq (np.ndarray): 時刻配列 (T,)
            u_seq_n (np.ndarray): 入力系列配列 (T, n_in)
            x_seq_n (AbstractState2DArray): 状態系列配列 (T, n_state)
            y_true_seq_n (np.ndarray): 真値の出力系列配列 (T, n_out)
            y_pred_seq_n (np.ndarray): 予測値の出力系列配列 (T, n_out)
            n_washout (int): washout区間の長さ
            readonly_without_copy (bool, optional): コピーせずにreadonly化する場合はTrue。デフォルトはFalse。
        """
        self._t_seq = t_seq
        self._u_seq_n = u_seq_n
        self._x_seq_n = x_seq_n
        self._y_true_seq_n = y_true_seq_n
        self._y_pred_seq_n = y_pred_seq_n
        self._n_washout = n_washout

        self._validate_and_finalize(readonly_without_copy=readonly_without_copy)

    def __len__(self):
        return len(self._t_seq)

    @property
    def n_in(self) -> int:
        return self._u_seq_n.shape[1]

    @property
    def n_state(self) -> int:
        return self._x_seq_n.shape[1]

    @property
    def n_out(self) -> int:
        return self._y_true_seq_n.shape[1]

    def _validate_and_finalize(self, *, readonly_without_copy: bool):
        # 型チェック
        assert isinstance(self._t_seq, np.ndarray), (type(self._t_seq), self._t_seq)
        assert isinstance(self._u_seq_n, np.ndarray), (type(self._u_seq_n), self._u_seq_n)
        assert isinstance(self._x_seq_n, AbstractState2DArray), (type(self._x_seq_n), self._x_seq_n)
        assert isinstance(self._y_true_seq_n, np.ndarray), \
            (type(self._y_true_seq_n), self._y_true_seq_n)
        assert isinstance(self._y_pred_seq_n, np.ndarray), \
            (type(self._y_pred_seq_n), self._y_pred_seq_n)
        assert isinstance(self._n_washout, int), (type(self._n_washout), self._n_washout)

        _len = len(self._t_seq)
        _n_in = self._u_seq_n.shape[1]
        _n_state = self._x_seq_n.shape[1]
        _n_out = self._y_true_seq_n.shape[1]

        expected_t_seq_shape = (_len,)
        assert self._t_seq.shape == expected_t_seq_shape, \
            (self._t_seq.shape, expected_t_seq_shape)
        expected_u_seq_n_shape = (_len, _n_in)
        assert self._u_seq_n.shape == expected_u_seq_n_shape, \
            (self._u_seq_n.shape, expected_u_seq_n_shape)
        expected_x_seq_n_shape = (_len, _n_state)
        assert self._x_seq_n.shape == expected_x_seq_n_shape, \
            (self._x_seq_n.shape, expected_x_seq_n_shape)
        expected_y_true_seq_n_shape = (_len, _n_out)
        assert self._y_true_seq_n.shape == expected_y_true_seq_n_shape, \
            (self._y_true_seq_n.shape, expected_y_true_seq_n_shape)
        expected_y_pred_seq_n_shape = (_len, _n_out)
        assert self._y_pred_seq_n.shape == expected_y_pred_seq_n_shape, \
            (self._y_pred_seq_n.shape, expected_y_pred_seq_n_shape)

        # コピーしてreadonly化
        if not readonly_without_copy:
            self._t_seq = self._t_seq.copy()
            self._u_seq_n = self._u_seq_n.copy()
            self._y_true_seq_n = self._y_true_seq_n.copy()
            self._y_pred_seq_n = self._y_pred_seq_n.copy()
        self._t_seq.setflags(write=False)
        self._u_seq_n.setflags(write=False)
        self._y_true_seq_n.setflags(write=False)
        self._y_pred_seq_n.setflags(write=False)
        # x_seq_nはAbstractState2DArrayなのでimmutable前提

        # washoutマスク作成
        self._washout_mask = np.zeros(_len, bool)
        self._washout_mask[self._n_washout:] = True
        self._washout_mask.setflags(write=False)

    @property
    def t_seq(self) -> np.ndarray:
        return self._t_seq

    @property
    def u_seq_n(self) -> np.ndarray:
        return self._u_seq_n

    @property
    def x_seq_n(self) -> AbstractState2DArray:
        return self._x_seq_n

    @property
    def y_true_seq_n(self) -> np.ndarray:
        return self._y_true_seq_n

    @property
    def y_pred_seq_n(self) -> np.ndarray:
        return self._y_pred_seq_n

    @property
    def n_washout(self) -> int:
        return self._n_washout

    @property
    def washout_mask(self) -> np.ndarray:
        return self._washout_mask

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
        return score_func(y_true=self.y_true_seq_n, y_pred=self.y_pred_seq_n)


class PredictionResultSet:
    def __init__(self, results: list[PredictionResult]):
        self._results = results

    def __len__(self):
        return len(self._results)

    def __getitem__(self, sample_index: int):
        return self._results[sample_index]

    # TODO: aggregated_score must return float
    @cache
    def aggregated_score(
            self,
            score: ScoreName,
            washout=True,
    ) -> np.ndarray:
        """
        結果全体のデータをひとつのデータとして集約し、scoreを計算する。
        """
        if washout:
            y_true_seq = np.array([r.y_true_seq_n[r.washout_mask] for r in self._results])
            y_pred_seq = np.array([r.y_pred_seq_n[r.washout_mask] for r in self._results])
        else:
            y_true_seq = np.array([r.y_true_seq_n for r in self._results])
            y_pred_seq = np.array([r.y_pred_seq_n for r in self._results])
        score_func = score_func_by_name(score, "2d")
        return score_func(y_true=y_true_seq, y_pred=y_pred_seq)
