from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.seed_or_rng import check_seed_or_rng_and_get_rng
from model.dataset import Discrete, Continuous, Dataset

"""
データセットの整理

時刻t_arr -> 入力（離散：u_arr, 連続：u_t(t)） -> 出力y_arr

時刻の生成
 - 時刻データのk番目はひとつのデータ要素の区間[t_k, t_(k+1))のt_kを表す
t_arr = generate_time_series(t_max, t_delta)  # [0, t_max)をt_deltaの間隔で分割
      = generate_time_series(t_max, n)  # [0, t_max)をn個に分割 [0, Δ, 2Δ, 3Δ, ..., t_max - Δ]

time-multiplexing用のデータは
 - FN: 各ステップの初めの入力と出力をその区間全体で引きずる（ステップの始めから終わりまで固定値をとる）
 - NV: 純粋な連続データが必要

ESN <- u_arr, y_arr 離散
FN <- u_arr, y_arr 離散
CD <- u_arr, y_arr 離散
NV <- t, u_t(t), y_t(t) 連続

任意の離散バージョン -> 連続バージョン の変換は可能
純粋な連続バージョンが欲しい場合もある
時刻は連続バージョンのときに必要

離散 DelayedSine -> 純粋な連続バージョンが必要
 - u_arr = sin wave, y_arr[k] = u_arr[k - lag]
離散 DelayedRandom
 - u_arr = U(0, 1), y_arr[k] = u_arr[k - lag]
離散 DelayedParityCheck
 - u_arr = {0, 1}, y_arr[k] = (sum{i = 1 to w} u_arr[k - i]) mod 2
離散 MackeyGlass -> 純粋な連続バージョンが必要
 - t = ..., u_t(0) = u_0, u_t(t) = 微分方程式により定まる, y_t(t) = u(t + t_step)
 - t_arr[k] = k * t_step, u_arr[k] = u_t(k * t_step), y_arr[k] = u_arr[k + 1]
離散の連続バージョン
 - t_step = ..., t = [0, 離散の要素数n * t_step) 
 - u_t(t) = u_arr[floor(t / t_step)], y_t(t) = y_arr[floor(t / t_step)]
"""


class AbstractDatasetGenerator(ABC):
    def __init__(self, *, t_max: float, t_step: float):
        self._t_max = t_max
        self._t_step = t_step

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        return dict(
            t_max=self._t_max,
            t_step=self._t_step,
        )

    @property
    def t_arr(self) -> Discrete:
        return np.arange(0, self._t_max, self._t_step)

    def __len__(self) -> int:
        return len(self.t_arr)

    @property
    @abstractmethod
    def u_arr(self) -> Discrete:
        raise NotImplementedError()

    @property
    @abstractmethod
    def y_arr(self) -> Discrete:
        raise NotImplementedError()

    @property
    @abstractmethod
    def u_t(self) -> Continuous:
        raise NotImplementedError()

    @property
    @abstractmethod
    def y_t(self) -> Continuous:
        raise NotImplementedError()

    def create(self) -> Dataset:
        return Dataset(
            name=self.name,
            parameters=self.parameters,
            t_arr=self.t_arr,
            u_arr=self.u_arr,
            y_arr=self.y_arr,
            u_t=self.u_t,
            y_t=self.y_t,
        )


class AbstractContinuousDatasetGenerator(AbstractDatasetGenerator, ABC):
    @property
    def u_arr(self) -> Discrete:
        return self.u_t(self.t_arr)

    @property
    def y_arr(self) -> Discrete:
        return self.y_t(self.t_arr)


class AbstractDiscreteDatasetGenerator(AbstractDatasetGenerator, ABC):
    @property
    def u_t(self) -> Continuous:
        def u_t(t: Discrete) -> Discrete:
            idx = np.floor(np.asarray(t) / self._t_step + 1e-11).astype(int)
            return self.u_arr[idx]

        return u_t

    @property
    def y_t(self) -> Continuous:
        def y_t(t: Discrete) -> Discrete:
            idx = np.floor(np.asarray(t) / self._t_step + 1e-11).astype(int)
            return self.y_arr[idx]

        return y_t


class DelayedSineDatasetGenerator(AbstractContinuousDatasetGenerator):  # 本質的に連続的なデータセット
    def __init__(
            self,
            *,
            t_max: float,
            t_step: float,
            freq: float,
            phase_offset: float,
            discrete_lag: float,
            amplitude: float,
    ):
        super().__init__(t_max=t_max, t_step=t_step)

        self._freq = freq
        self._phase_offset = phase_offset
        self._discrete_lag = discrete_lag
        self._amplitude = amplitude

    @property
    def name(self) -> str:
        return f"DelayedSine({self._discrete_lag})"

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(
            **super().parameters,
            freq=self._freq,
            phase_offset=self._phase_offset,
            discrete_lag=self._discrete_lag,
            amplitude=self._amplitude,
        )

    def __f_t(self, t: Discrete) -> Discrete:
        return self._amplitude * np.sin(2 * np.pi * self._freq * t + self._phase_offset)

    @property
    def u_t(self) -> Continuous:
        def u_t(t: Discrete) -> Discrete:
            return self.__f_t(t)

        return u_t

    @property
    def y_t(self) -> Continuous:
        def y_t(t: Discrete) -> Discrete:
            return self.__f_t(t - self._discrete_lag * self._t_step)

        return y_t


class NoisyDelayedSineDatasetGenerator(DelayedSineDatasetGenerator):
    def __init__(
            self,
            *,
            t_max: float,
            t_step: float,
            freq: float,
            phase_offset: float,
            discrete_lag: float,
            amplitude: float,
            noise_std: float,
            seed: int | None = None,
            rng: np.random.RandomState | None = None,
    ):
        super().__init__(
            t_max=t_max,
            t_step=t_step,
            freq=freq,
            phase_offset=phase_offset,
            discrete_lag=discrete_lag,
            amplitude=amplitude,
        )

        self._noise_std = noise_std
        self._rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)

    @property
    def name(self) -> str:
        return f"NoisyDelayedSine({self._discrete_lag}, {self._noise_std})"

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(
            **super().parameters,
            noise_std=self._noise_std,
        )

    @property
    def u_t(self) -> Continuous:  # 生成の度に異なるノイズが加わる
        super_u_t = super().u_t

        def u_t(t: Discrete) -> Discrete:
            a = super_u_t(t)
            if not isinstance(t, np.ndarray):
                r = float(a + self._rng.normal(0, self._noise_std))
            else:
                r = a + self._rng.normal(0, self._noise_std, len(a))
            return r

        return u_t

    @property
    def y_t(self) -> Continuous:  # 生成の度に異なるノイズが加わる
        super_y_t = super().y_t

        def y_t(t: Discrete) -> Discrete:
            a = super_y_t(t)
            if not isinstance(t, np.ndarray):
                r = float(a + self._rng.normal(0, self._noise_std))
            else:
                r = a + self._rng.normal(0, self._noise_std, len(a))
            return r

        return y_t


class DelayedRandomDatasetGenerator(AbstractDiscreteDatasetGenerator):  # 本質的に離散的なデータセット
    def __init__(
            self,
            *,
            t_max: float,
            t_step: float,
            discrete_lag: float,
            low: float,
            high: float,
            seed: int | None = None,
            rng: np.random.RandomState | None = None,
    ):
        super().__init__(t_max=t_max, t_step=t_step)

        self._discrete_lag = discrete_lag
        self._low = low
        self._high = high
        self._rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)

        self._series = self._rng.uniform(self._low, self._high, len(self.t_arr))

    @property
    def name(self) -> str:
        return f"DelayedRandom({self._discrete_lag})"

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(
            t_max=self._t_max,
            t_step=self._t_step,
            discrete_lag=self._discrete_lag,
            low=self._low,
            high=self._high,
        )

    @property
    def u_arr(self) -> Discrete:
        return self._series

    @property
    def y_arr(self) -> Discrete:
        y_arr = np.zeros_like(self.u_arr)
        y_arr[self._discrete_lag:] = self.u_arr[:-self._discrete_lag]
        return y_arr


class ParityCheckDatasetGenerator(AbstractDiscreteDatasetGenerator):  # 本質的に離散的なデータセット
    def __init__(
            self,
            *,
            t_max: float,
            t_step: float,
            window_size: int,
            seed: int | None = None,
            rng: np.random.RandomState | None = None,
    ):
        super().__init__(t_max=t_max, t_step=t_step)
        self._window_size = window_size
        self._rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)

        self._series = self._rng.choice([0, 1], size=len(self.t_arr))

    @property
    def name(self) -> str:
        return f"ParityCheck({self._window_size})"

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(
            **super().parameters,
            window_size=self._window_size,
        )

    @property
    def u_arr(self) -> Discrete:
        return self._series

    @property
    def y_arr(self) -> Discrete:
        y_arr = np.zeros_like(self.u_arr)
        for k in range(self._window_size, len(self.u_arr)):
            y_arr[k] = np.sum(self.u_arr[k - self._window_size:k]) % 2
        return y_arr


class NarmaDatasetGenerator(AbstractDiscreteDatasetGenerator):
    """
    NARMAタスクのデータセット。
    本質的に離散的なデータセット。
    u_arr: 入力系列（ランダム一様分布）
    y_arr: NARMA方程式で生成される出力系列
    """

    def __init__(
            self,
            *,
            t_max: float,
            t_step: float,
            n: int = 10,
            a: float = 0.3,
            b: float = 0.05,
            c: float = 1.5,
            d: float = 0.1,
            low: float = 0.0,
            high: float = 0.5,
            seed: int | None = None,
            rng: np.random.RandomState | None = None,
    ):
        super().__init__(t_max=t_max, t_step=t_step)
        self._n = n
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._low = low
        self._high = high
        self._rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
        self._series = self._rng.uniform(self._low, self._high, len(self.t_arr))

    @property
    def name(self) -> str:
        return f"Narma({self._n})"

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(
            **super().parameters,
            n=self._n,
            a=self._a,
            b=self._b,
            c=self._c,
            d=self._d,
            low=self._low,
            high=self._high,
        )

    @property
    def u_arr(self) -> Discrete:
        return self._series

    @property
    def y_arr(self) -> Discrete:
        y_arr = np.zeros_like(self.u_arr)
        n = self._n
        a = self._a
        b = self._b
        c = self._c
        d = self._d
        u = self.u_arr
        for k in range(n - 1, len(u) - 1):
            y_arr[k + 1] = (
                    a * y_arr[k]
                    + b * y_arr[k] * np.sum(y_arr[k - n + 1:k + 1])
                    + c * u[k] * u[k - n + 1]
                    + d
            )
        return y_arr


def train_test_split(
        u_arr, y_arr, t_arr=None,
        /,
        test_ratio=0.3,
        n_washout: int | None = None,
        n_train_washout: int | None = None,
        n_test_washout: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
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
        t_arr (np.ndarray): 時刻データ配列（省略可）
        test_ratio (float): train/test分割比率 (0.0-1.0, testの割合)
        n_washout (int|None): train/test両方のwashout数
        n_train_washout (int|None): 先頭のwashout数
        n_test_washout (int|None): trainとtestの間のwashout数

    Returns:
        tuple: (u_train, u_test), (y_train, y_test), (t_train, t_test)
        ただし時刻データ配列未指定なら(u_train, u_test), (y_train, y_test)
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
    if t_arr is not None:
        t_train = t_arr[train_start:train_end]
        t_test = t_arr[test_start:test_end]
        return (u_train, u_test), (y_train, y_test), (t_train, t_test)
    return (u_train, u_test), (y_train, y_test)


def _preview_dataset():
    t_max = 5
    t_step = 0.1
    discrete_lag = 3
    noise_std = 0.05

    ds_delayed_sine = DelayedSineDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        freq=1,
        phase_offset=np.pi / 2,
        discrete_lag=discrete_lag,
        amplitude=1,
    )

    ds_noisy_delayed_sine = NoisyDelayedSineDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        freq=1,
        phase_offset=np.pi / 2,
        discrete_lag=discrete_lag,
        amplitude=1,
        noise_std=noise_std,
    )

    ds_delayed_random = DelayedRandomDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        discrete_lag=discrete_lag,
        low=0,
        high=1,
    )

    ds_parity_check = ParityCheckDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        window_size=discrete_lag,
    )

    ds_lst = [
        (f"DelayedSine({discrete_lag})", ds_delayed_sine),
        (f"NoisyDelayedSine({discrete_lag}, {noise_std})", ds_noisy_delayed_sine),
        (f"DelayedRandom({discrete_lag})", ds_delayed_random),
        (f"ParityCheck({discrete_lag})", ds_parity_check),
    ]

    from matplotlib import pyplot as plt

    t_step_continuous = 0.002
    t_arr_continuous = np.arange(0, t_max, t_step_continuous)

    for name, ds in ds_lst:
        plt.figure(figsize=(10, 5))
        # discrete
        plt.subplot(2, 1, 1)
        plt.plot(ds.t_arr, ds.u_arr, label=f"{name} (u)", marker="o")
        plt.plot(ds.t_arr, ds.y_arr, label=f"{name} (y)", marker="o")
        plt.xlabel("t")
        plt.ylabel("u, y")
        plt.grid()
        plt.title("Discrete")
        plt.legend()
        # continuous
        plt.subplot(2, 1, 2)
        plt.plot(t_arr_continuous, ds.u_t(t_arr_continuous), label=f"{name} (u_continuous)")
        plt.plot(t_arr_continuous, ds.y_t(t_arr_continuous), label=f"{name} (y_continuous)")
        plt.xlabel("t")
        plt.ylabel("u, y")
        plt.grid()
        plt.title("Continuous")
        plt.legend()
        # show
        plt.suptitle(name)
        plt.show()


if __name__ == '__main__':
    _preview_dataset()
