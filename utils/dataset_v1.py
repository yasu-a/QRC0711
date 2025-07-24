# from abc import ABC, abstractmethod
# from typing import Callable
#
# import numpy as np
#
# from utils.seed_or_rng import check_seed_or_rng_and_get_rng


# class InputSignalMappedDatasetGenerator(ABC):
#     """
#     入力信号（u_arr）を受けて (u_arr, y_arr) のタプルを生成するデータセットの抽象基底クラス。
#     """
#
#     @abstractmethod
#     def __call__(self, u_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         """Generate output data for given input array.
#
#         Args:
#             u_arr (np.ndarray): Input signal array
#
#         Returns:
#             tuple[np.ndarray, np.ndarray]: Tuple containing input array and corresponding output array
#         """
#         raise NotImplementedError()
#
#     @classmethod
#     def create(cls, u_arr: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
#         """Create dataset from input array using provided parameters.
#
#         Args:
#             u_arr (np.ndarray): Input signal array
#             **kwargs: Additional arguments passed to class constructor
#
#         Returns:
#             tuple[np.ndarray, np.ndarray]: Tuple containing input array and corresponding output array
#         """
#         return cls(**kwargs)(u_arr)  # type: ignore


# class RandomInputSignalMappedDatasetGenerator(InputSignalMappedDatasetGenerator, ABC):
#     @classmethod
#     def create_random_input(cls, *, n: int, seed=None, rng=None, low=0.0, high=0.5) -> np.ndarray:
#         """Generate random input data.
#
#         Args:
#             n (int): Number of data points to generate
#             seed (int, optional): Random seed for reproducibility. Defaults to None.
#             rng (np.random.RandomState, optional): Random number generator. Defaults to None.
#             low (float, optional): Lower bound of uniform distribution. Defaults to 0.0.
#             high (float, optional): Upper bound of uniform distribution. Defaults to 0.5.
#
#         Returns:
#             np.ndarray: Random input array
#         """
#         rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
#         return rng.uniform(low, high, n)
#
#     @classmethod
#     def create_with_random_input(cls, *, n: int, seed=None, rng=None, low=0.0, high=0.5, **kwargs) \
#             -> tuple[np.ndarray, np.ndarray]:
#         """Generate random input data and corresponding output.
#
#         Args:
#             n (int): Number of data points to generate
#             seed (int, optional): Random seed for reproducibility. Defaults to None.
#             rng (np.random.RandomState, optional): Random number generator. Defaults to None.
#             low (float, optional): Lower bound of uniform distribution. Defaults to 0.0.
#             high (float, optional): Upper bound of uniform distribution. Defaults to 0.5.
#             **kwargs: Additional arguments passed to class constructor
#
#         Returns:
#             tuple[np.ndarray, np.ndarray]: Random input array and corresponding output array
#         """
#         u_arr = cls.create_random_input(n=n, seed=seed, rng=rng, low=low, high=high)
#         return cls.create(u_arr, **kwargs)


# class TimeMappedDatasetGenerator(ABC):
#     """
#     時刻配列（t_arr）を受けて (u_arr, y_arr) のタプルを生成するデータセットの抽象基底クラス。
#     """
#
#     @abstractmethod
#     def __call__(self, t_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         """Generate input/output data for given time array.
#
#         Args:
#             t_arr (np.ndarray): Time points array
#
#         Returns:
#             tuple[np.ndarray, np.ndarray]: Tuple containing input array and corresponding output array
#         """
#         raise NotImplementedError()
#
#     @classmethod
#     def create(cls, t_arr: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
#         """Create dataset from time array using provided parameters.
#
#         Args:
#             t_arr (np.ndarray): Time points array
#             **kwargs: Additional arguments passed to class constructor
#
#         Returns:
#             tuple[np.ndarray, np.ndarray]: Tuple containing input array and corresponding output array
#         """
#         return cls(**kwargs)(t_arr)  # type: ignore


# class Narma(RandomInputSignalMappedDatasetGenerator):
#     """
#     NARMAタスクのデータ生成クラス。
#     入力u_arrから (u_arr, y_arr) を返す。
#     """
#
#     def __init__(self, n: int = 10, a=0.3, b=0.05, c=1.5, d=0.1):
#         self.n = n
#         self.a = a
#         self.b = b
#         self.c = c
#         self.d = d
#
#     def __call__(self, u_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         y_arr = np.zeros_like(u_arr)
#         for k in range(self.n - 1, len(u_arr) - 1):
#             y_arr[k + 1] = (
#                     self.a * y_arr[k]
#                     + self.b * y_arr[k] * np.sum(y_arr[k - self.n + 1:k + 1])
#                     + self.c * u_arr[k] * u_arr[k - self.n + 1]
#                     + self.d
#             )
#         return u_arr, y_arr


# class DelayedSine(TimeMappedDatasetGenerator):
#     """
#     サイン波＋遅延出力データセット。
#     t_arrから (u_arr, y_arr) を返す。
#     u_arr: サイン波＋ノイズ
#     y_arr: u_arrをt_lagだけ遅延させたもの
#     """
#
#     def __init__(self, noise_std: float = 0.01, freq: float = 1.0, seed: int = None,
#                  rng: np.random.RandomState = None,
#                  lag: int = 1):
#         self.noise_std = noise_std
#         self.freq = freq
#         self.lag = lag
#         self.rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
#
#     def __call__(self, t_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         u_arr = np.sin(2 * np.pi * self.freq * t_arr + self.rng.uniform(0, np.pi))
#         y_arr = np.zeros_like(u_arr)
#         y_arr[self.lag:] = u_arr[:-self.lag]
#         u_arr += self.rng.normal(0, self.noise_std, len(t_arr))
#         y_arr += self.rng.normal(0, self.noise_std, len(t_arr))
#         return u_arr, y_arr


# class LaggedInput(RandomInputSignalMappedDatasetGenerator):
#     """
#     入力信号を遅延させたデータセット。
#     """
#
#     def __init__(self, lag: int = 1):
#         self.lag = lag
#
#     def __call__(self, u_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         y_arr = np.zeros_like(u_arr)
#         y_arr[self.lag:] = u_arr[:-self.lag]
#         return u_arr, y_arr


# def train_test_split(
#         u_arr, y_arr, t_arr=None,
#         /,
#         test_ratio=0.3,
#         n_washout: int | None = None,
#         n_train_washout: int | None = None,
#         n_test_washout: int | None = None,
# ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
#     """
#     データを訓練・テストに分割する関数。
#
#     - 先頭からn_train_washout個分を除外
#     - trainとtestの間にn_test_washout個分のwashoutを挟む
#     - n_washoutを指定した場合はn_train_washout, n_test_washout両方に同じ値をセット
#     - どれも指定しない場合はwashoutなし
#     - n_washoutとn_train_washout/n_test_washoutを同時指定はエラー
#     - 残りをtest_ratioの比率でtrain/testに分割
#
#     Args:
#         u_arr (np.ndarray): 入力データ配列
#         y_arr (np.ndarray): 出力データ配列
#         t_arr (np.ndarray): 時刻データ配列（省略可）
#         test_ratio (float): train/test分割比率 (0.0-1.0, testの割合)
#         n_washout (int|None): train/test両方のwashout数
#         n_train_washout (int|None): 先頭のwashout数
#         n_test_washout (int|None): trainとtestの間のwashout数
#
#     Returns:
#         tuple: (u_train, y_train, u_test, y_test, t_train, t_test)
#         ただし時刻データ配列未指定なら(u_train, y_train, u_test, y_test)
#     """
#     if len(u_arr) != len(y_arr):
#         raise ValueError("u_arr and y_arr must have the same length")
#
#     # washout指定の解釈
#     if n_washout is not None:
#         if n_train_washout is not None or n_test_washout is not None:
#             raise ValueError(
#                 "n_washoutとn_train_washout/n_test_washoutを同時に指定することはできません")
#         n_train_washout = n_washout
#         n_test_washout = n_washout
#     else:
#         if n_train_washout is None:
#             n_train_washout = 0
#         if n_test_washout is None:
#             n_test_washout = 0
#
#     n_total = len(u_arr)
#     n_remain = n_total - n_train_washout - n_test_washout
#     n_test = int(n_remain * test_ratio)
#     n_train = n_remain - n_test
#
#     assert n_train >= 1, f"n_train={n_train} が1未満です。データ数や比率を見直してください。"
#     assert n_test >= 1, f"n_test={n_test} が1未満です。データ数や比率を見直してください。"
#
#     train_start = n_train_washout
#     train_end = train_start + n_train
#     test_start = train_end + n_test_washout
#     test_end = test_start + n_test
#
#     u_train = u_arr[train_start:train_end]
#     y_train = y_arr[train_start:train_end]
#     u_test = u_arr[test_start:test_end]
#     y_test = y_arr[test_start:test_end]
#     if t_arr is not None:
#         t_train = t_arr[train_start:train_end]
#         t_test = t_arr[test_start:test_end]
#         return (u_train, u_test), (y_train, y_test), (t_train, t_test)
#     return (u_train, u_test), (y_train, y_test)


# def to_continuous_function(f_arr: np.ndarray, *, delta_t: float) -> Callable[[float], float]:
#     """
#     配列を連続関数に変換する関数。
#     """
#
#     def func(t: float) -> float:
#         i = int(t // delta_t)
#         try:
#             return float(f_arr[i])
#         except IndexError:
#             raise ValueError(
#                 f"t={t:.5f}, t_step={delta_t:.5f} generates an array index out of range: {i}, "
#                 f"expected index less than {len(f_arr)}"
#             )
#
#     return func


# def to_continuous_function_with_linspace_time(t_arr, f_arr) -> Callable[[float], float]:
#     delta_t = np.diff(t_arr).mean()
#     u_t = to_continuous_function(f_arr, delta_t=delta_t)
#     assert np.allclose(
#         np.vectorize(u_t)(t_arr + delta_t / 2),
#         f_arr,
#     )
#     return u_t
