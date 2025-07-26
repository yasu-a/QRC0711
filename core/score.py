from typing import Literal, Callable

import numpy as np
from sklearn.metrics import r2_score as _sklearn_r2_score

"""
shape == (サンプル数, 時系列数, 特徴量数): 時系列数に渡ってスコアを計算し(サンプル数, 特徴量数)を返す
shape == (時系列数, 特徴量数): 時系列数に渡ってスコアを計算し(特徴量数)を返す
shape == (時系列数,): 時系列数に渡ってスコアを計算しスカラーを返す
1dの実装はsklearnかnumpyを使う
"""


def _apply_score_1d(score_1d_func):
    def wrapper(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        1次元配列のy_trueとy_predのスコアを計算するラッパー関数。
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"`y_true` and `y_pred` must have the same shape. "
                f"y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}"
            )
        if y_true.ndim != 1:
            raise ValueError(
                f"Invalid shape: {y_true.shape}. "
                f"`y_true.shape` must be (n_timesteps,) for 1D score calculation."
            )
        return score_1d_func(y_true=y_true, y_pred=y_pred)

    return wrapper


def _apply_score_2d(score_1d_func):
    def wrapper(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        2次元配列のy_trueとy_predのスコアを計算するラッパー関数。
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"`y_true` and `y_pred` must have the same shape. "
                f"y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}"
            )
        if y_true.ndim != 2:
            raise ValueError(
                f"Invalid shape: {y_true.shape}. "
                f"`y_true.shape` must be (n_timesteps, n_features) for 2D score calculation."
            )
        return np.array([
            score_1d_func(y_true=y_true[:, i_feature], y_pred=y_pred[:, i_feature])
            for i_feature in range(y_true.shape[1])
        ])

    return wrapper


def _apply_score_3d(score_1d_func):
    def wrapper(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        3次元配列のy_trueとy_predのスコアを計算するラッパー関数。
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"`y_true` and `y_pred` must have the same shape. "
                f"y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}"
            )
        if y_true.ndim != 3:
            raise ValueError(
                f"Invalid shape: {y_true.shape}. "
                f"`y_true.shape` must be (n_samples, n_timesteps, n_features) for 3D score calculation."
            )
        return np.array([
            [
                score_1d_func(
                    y_true=y_true[i_sample, :, i_feature],
                    y_pred=y_pred[i_sample, :, i_feature],
                )
                for i_feature in range(y_true.shape[2])
            ]
            for i_sample in range(y_true.shape[0])
        ])

    return wrapper


def _r2_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_true, y_predに対してR^2スコアを計算する。
    """
    return _sklearn_r2_score(y_true, y_pred)


@_apply_score_1d
def r2_score_1d(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_trueとy_predのR^2スコアを計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_timesteps,)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        float: R^2スコア（スカラー値）。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または1次元でない場合。
    """
    return _r2_score_1d(y_true, y_pred)


@_apply_score_2d
def r2_score_2d(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    2次元配列のy_trueとy_predのR^2スコアを計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_timesteps, n_features)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray: 各特徴量ごとのR^2スコア配列。shape は (n_features,)。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または2次元でない場合。
    """
    # noinspection PyTypeChecker
    return _r2_score_1d(y_true, y_pred)


@_apply_score_3d
def r2_score_3d(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    3次元配列のy_trueとy_predのR^2スコアを計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_samples, n_timesteps, n_features)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray: 各サンプル・各特徴量ごとのR^2スコア配列。shape は (n_samples, n_features)。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または3次元でない場合。
    """
    # noinspection PyTypeChecker
    return _r2_score_1d(y_true, y_pred)


def _capacity_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_true, y_predに対してcapacityスコア（相関係数の2乗）を計算する。
    標準偏差が0の場合は0.0を返す。
    """
    cov = np.cov(y_true, y_pred, ddof=0)[0, 1]
    std1 = np.std(y_true)
    std2 = np.std(y_pred)
    if std1 == 0 or std2 == 0:
        return 0.0
    return (cov ** 2) / (std1 ** 2 * std2 ** 2)


@_apply_score_1d
def capacity_score_1d(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_trueとy_predのcapacityスコア（相関係数の2乗）を計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_timesteps,)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        float: capacityスコア（スカラー値）。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または1次元でない場合。
    """
    return _capacity_score_1d(y_true, y_pred)


@_apply_score_2d
def capacity_score_2d(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    2次元配列のy_trueとy_predのcapacityスコア（相関係数の2乗）を計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_timesteps, n_features)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray: 各特徴量ごとのcapacityスコア配列。shape は (n_features,)。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または2次元でない場合。
    """
    # noinspection PyTypeChecker
    return _capacity_score_1d(y_true, y_pred)


@_apply_score_3d
def capacity_score_3d(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    3次元配列のy_trueとy_predのcapacityスコア（相関係数の2乗）を計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_samples, n_timesteps, n_features)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray: 各サンプル・各特徴量ごとのcapacityスコア配列。shape は (n_samples, n_features)。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または3次元でない場合。
    """
    # noinspection PyTypeChecker
    return _capacity_score_1d(y_true, y_pred)


def _mse_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_true, y_predに対して平均二乗誤差（MSE）を計算する。
    """
    return float(np.mean((y_true - y_pred) ** 2))


@_apply_score_1d
def mse_score_1d(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_trueとy_predの平均二乗誤差（MSE）を計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_timesteps,)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        float: MSE（スカラー値）。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または1次元でない場合。
    """
    return _mse_score_1d(y_true, y_pred)


@_apply_score_2d
def mse_score_2d(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    2次元配列のy_trueとy_predの平均二乗誤差（MSE）を計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_timesteps, n_features)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray: 各特徴量ごとのMSE配列。shape は (n_features,)。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または2次元でない場合。
    """
    # noinspection PyTypeChecker
    return _mse_score_1d(y_true, y_pred)


@_apply_score_3d
def mse_score_3d(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    3次元配列のy_trueとy_predの平均二乗誤差（MSE）を計算するユーティリティ関数。

    Args:
        y_true (np.ndarray): 正解値配列。shape は (n_samples, n_timesteps, n_features)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray: 各サンプル・各特徴量ごとのMSE配列。shape は (n_samples, n_features)。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、または3次元でない場合。
    """
    # noinspection PyTypeChecker
    return _mse_score_1d(y_true, y_pred)


ScoreName = Literal["r2", "capacity", "mse"]
DimensionName = Literal["1d", "2d", "3d"]


def score_func_by_name(name: ScoreName, dim: DimensionName) -> Callable:
    """
    スコア名と次元から対応するスコア関数を返す。

    Args:
        name (ScoreName): スコア名（"r2", "capacity", "mse"）
        dim (DimensionName): 次元名（"1d", "2d", "3d"）

    Returns:
        Callable: 対応するスコア関数

    Raises:
        ValueError: 無効なスコア名または次元名の場合
    """
    # スコア関数のマッピング
    score_map = {
        "r2": {"1d": r2_score_1d, "2d": r2_score_2d, "3d": r2_score_3d},
        "capacity": {"1d": capacity_score_1d, "2d": capacity_score_2d, "3d": capacity_score_3d},
        "mse": {"1d": mse_score_1d, "2d": mse_score_2d, "3d": mse_score_3d},
    }

    if name not in score_map:
        raise ValueError(f"Invalid score name: {name}")

    if dim not in score_map[name]:
        raise ValueError(f"Invalid dimension name: {dim}")

    return score_map[name][dim]
