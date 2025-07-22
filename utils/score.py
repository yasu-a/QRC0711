from typing import Literal, Callable

import numpy as np
from sklearn.metrics import r2_score as _sklearn_r2_score

"""
shape == (サンプル数, 時系列数, 特徴量数): 時系列数に渡ってスコアを計算し(サンプル数, 特徴量数)を返す
shape == (時系列数, 特徴量数): 時系列数に渡ってスコアを計算し(特徴量数)を返す
shape == (時系列数,): 時系列数に渡ってスコアを計算しスカラーを返す
1dの実装はsklearnかnumpyを使う
"""


def apply_score_nd(score_1d_func):
    def wrapper(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        y_trueとy_predのスコアを計算する多次元対応ラッパー関数。
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"`y_true` and `y_pred` must have the same shape. "
                f"y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}"
            )
        if y_true.ndim == 3:
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
        elif y_true.ndim == 2:
            return np.array([
                score_1d_func(y_true=y_true[:, i_feature], y_pred=y_pred[:, i_feature])
                for i_feature in range(y_true.shape[1])
            ])
        elif y_true.ndim == 1:
            return score_1d_func(y_true=y_true, y_pred=y_pred)
        else:
            raise ValueError(
                f"Invalid shape: {y_true.shape}. "
                f"`y_true.shape` must be (n_samples, n_timesteps, n_features), (n_timesteps, n_features), or (n_timesteps)."
            )

    return wrapper


def _r2_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_true, y_predに対してR^2スコアを計算する。
    """
    return _sklearn_r2_score(y_true, y_pred)


@apply_score_nd
def r2_score(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    y_trueとy_predのR^2スコアを計算するユーティリティ関数。

    - shape == (サンプル数, 時系列数, 特徴量数): 各サンプル・各特徴量ごとに時系列方向でR^2スコアを計算し、(サンプル数, 特徴量数)の配列を返す。
    - shape == (時系列数, 特徴量数): 各特徴量ごとに時系列方向でR^2スコアを計算し、(特徴量数,)の配列を返す。
    - shape == (時系列数,): 1次元配列としてR^2スコア（スカラー値）を返す。

    Args:
        y_true (np.ndarray): 正解値配列。shapeは(サンプル数, 時系列数, 特徴量数)、(時系列数, 特徴量数)、または(時系列数,)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray | float: R^2スコア配列またはスカラー値。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、またはサポートされていない次元数の場合。
    """
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


@apply_score_nd
def capacity_score(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    y_trueとy_predのcapacityスコア（相関係数の2乗）を計算するユーティリティ関数。

    - shape == (サンプル数, 時系列数, 特徴量数): 各サンプル・各特徴量ごとに時系列方向でcapacityスコアを計算し、(サンプル数, 特徴量数)の配列を返す。
    - shape == (時系列数, 特徴量数): 各特徴量ごとに時系列方向でcapacityスコアを計算し、(特徴量数,)の配列を返す。
    - shape == (時系列数,): 1次元配列としてcapacityスコア（スカラー値）を返す。

    Args:
        y_true (np.ndarray): 正解値配列。shapeは(サンプル数, 時系列数, 特徴量数)、(時系列数, 特徴量数)、または(時系列数,)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray | float: capacityスコア配列またはスカラー値。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、またはサポートされていない次元数の場合。
    """
    return _capacity_score_1d(y_true, y_pred)


def _mse_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1次元配列のy_true, y_predに対して平均二乗誤差（MSE）を計算する。
    """
    return np.mean((y_true - y_pred) ** 2)


@apply_score_nd
def mse_score(*, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    y_trueとy_predの平均二乗誤差（MSE）を計算するユーティリティ関数。

    - shape == (サンプル数, 時系列数, 特徴量数): 各サンプル・各特徴量ごとに時系列方向でMSEを計算し、(サンプル数, 特徴量数)の配列を返す。
    - shape == (時系列数, 特徴量数): 各特徴量ごとに時系列方向でMSEを計算し、(特徴量数,)の配列を返す。
    - shape == (時系列数,): 1次元配列としてMSE（スカラー値）を返す。

    Args:
        y_true (np.ndarray): 正解値配列。shapeは(サンプル数, 時系列数, 特徴量数)、(時系列数, 特徴量数)、または(時系列数,)。
        y_pred (np.ndarray): 予測値配列。y_trueと同じshape。

    Returns:
        np.ndarray | float: MSE配列またはスカラー値。

    Raises:
        ValueError: y_trueとy_predのshapeが一致しない場合、またはサポートされていない次元数の場合。
    """
    return _mse_score_1d(y_true, y_pred)


ScoreName = Literal["r2", "capacity", "mse"]


def score_func_by_name(name: ScoreName) -> Callable:
    if name == "r2":
        return r2_score
    elif name == "capacity":
        return capacity_score
    elif name == "mse":
        return mse_score
    else:
        raise ValueError(f"Invalid score name: {name}")
