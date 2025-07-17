from abc import ABC, abstractmethod
from typing import Callable, Protocol

import numpy as np


class DatasetGeneratorProtocol(Protocol):
    def __call__(self, u_arr: np.ndarray) -> np.ndarray:
        ...

    @classmethod
    def create(cls, u_arr: np.ndarray, **kwargs) -> np.ndarray:
        ...


class DatasetGenerator(ABC):
    @abstractmethod
    def __call__(self, u_arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def create(cls, u_arr: np.ndarray, **kwargs):
        return cls(**kwargs)(u_arr)  # type: ignore


class RandomInputMixin(DatasetGeneratorProtocol):
    @classmethod
    def create_random_input(cls, *, n: int, seed=0, low=0.0, high=0.5) -> np.ndarray:
        """Generate random input data.

        Args:
            n (int): Number of data points to generate
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            low (float, optional): Lower bound of uniform distribution. Defaults to 0.0.
            high (float, optional): Upper bound of uniform distribution. Defaults to 0.5.

        Returns:
            np.ndarray: Random input array
        """
        np.random.seed(seed=seed)
        return np.random.uniform(low, high, n)

    @classmethod
    def create_with_random_input(cls, *, n: int, seed=0, low=0.0, high=0.5, **kwargs) \
            -> tuple[np.ndarray, np.ndarray]:
        """Generate random input data and corresponding output.

        Args:
            n (int): Number of data points to generate
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            low (float, optional): Lower bound of uniform distribution. Defaults to 0.0.
            high (float, optional): Upper bound of uniform distribution. Defaults to 0.5.

        Returns:
            tuple[np.ndarray, np.ndarray]: Random input array and corresponding output array
        """
        u_arr = cls.create_random_input(n=n, seed=seed, low=low, high=high)
        y_arr = cls.create(u_arr, **kwargs)
        return u_arr, y_arr


class Narma10(DatasetGenerator, RandomInputMixin):
    """
    NARMA10タスクのデータ生成クラス。

    数式:
        y_{k+1} = a y_k + b y_k \sum_{i=0}^9 y_{k-i} + c u_k u_{k-9} + d

    Args:
        a (float): y_kの係数
        b (float): y_kと過去のyの和の積の係数
        c (float): u_kとu_{k-9}の積の係数
        d (float): 定数項
    """

    def __init__(self, a=0.3, b=0.05, c=1.5, d=0.1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, u_arr: np.ndarray) -> np.ndarray:
        y_arr = np.zeros_like(u_arr)
        for k in range(9, len(u_arr) - 1):
            y_arr[k + 1] = (
                    self.a * y_arr[k]
                    + self.b * y_arr[k] * np.sum(y_arr[k - 9:k + 1])
                    + self.c * u_arr[k] * u_arr[k - 9]
                    + self.d
            )
        return y_arr


def train_test_split(
        u_arr, y_arr,
        *,
        test_ratio=0.3,
        n_washout: int | None = None,
        n_train_washout: int | None = None,
        n_test_washout: int | None = None,
):
    """
    データを訓練・テストに分割する関数。

    - 先頭からn_train_washout個分を除外
    - trainとtestの間にn_test_washout個分のwashoutを挟む
    - n_washoutを指定した場合はn_train_washout, n_test_washout両方に同じ値をセット
    - どれも指定しない場合はwashoutなし
    - n_washoutとn_train_washout/n_test_washoutを同時指定はエラー
    - 残りをtest_ratioの比率でtrain/testに分割

    Args:
        u_arr (np.ndarray): 入力データ配列
        y_arr (np.ndarray): 出力データ配列
        test_ratio (float): train/test分割比率 (0.0-1.0, testの割合)
        n_washout (int|None): train/test両方のwashout数
        n_train_washout (int|None): 先頭のwashout数
        n_test_washout (int|None): trainとtestの間のwashout数

    Returns:
        tuple: (u_train, y_train, u_test, y_test)
    """
    if len(u_arr) != len(y_arr):
        raise ValueError("u_arr and y_arr must have the same length")

    # washout指定の解釈
    if n_washout is not None:
        if n_train_washout is not None or n_test_washout is not None:
            raise ValueError(
                "n_washoutとn_train_washout/n_test_washoutを同時に指定することはできません")
        n_train_washout = n_washout
        n_test_washout = n_washout
    else:
        if n_train_washout is None:
            n_train_washout = 0
        if n_test_washout is None:
            n_test_washout = 0

    n_total = len(u_arr)
    n_remain = n_total - n_train_washout - n_test_washout
    n_test = int(n_remain * test_ratio)
    n_train = n_remain - n_test

    assert n_train >= 1, f"n_train={n_train} が1未満です。データ数や比率を見直してください。"
    assert n_test >= 1, f"n_test={n_test} が1未満です。データ数や比率を見直してください。"

    train_start = n_train_washout
    train_end = train_start + n_train
    test_start = train_end + n_test_washout
    test_end = test_start + n_test

    u_train = u_arr[train_start:train_end]
    y_train = y_arr[train_start:train_end]
    u_test = u_arr[test_start:test_end]
    y_test = y_arr[test_start:test_end]

    return u_train, y_train, u_test, y_test


def to_continuous_function(arr: np.ndarray, *, t_step: float) -> Callable[[float], float]:
    """
    配列を連続関数に変換する関数。
    """

    def func(t: float) -> float:
        i = int(t // t_step)
        try:
            return float(arr[i])
        except IndexError:
            raise ValueError(
                f"t={t:.5f}, t_step={t_step:.5f} generates an array index out of range: {i}, "
                f"expected index less than {len(arr)}"
            )

    return func
