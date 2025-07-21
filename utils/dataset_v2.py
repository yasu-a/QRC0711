from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from utils.seed_or_rng import check_seed_or_rng_and_get_rng

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

_Discrete = np.ndarray | float
_Continuous = Callable[[_Discrete], _Discrete]


class AbstractDataset(ABC):
    def __init__(self, *, t_max: float, t_step: float):
        self._t_max = t_max
        self._t_step = t_step

    @property
    def t_arr(self) -> _Discrete:
        return np.arange(0, self._t_max, self._t_step)

    def __len__(self) -> int:
        return len(self.t_arr)

    @property
    @abstractmethod
    def u_arr(self) -> _Discrete:
        raise NotImplementedError()

    @property
    @abstractmethod
    def y_arr(self) -> _Discrete:
        raise NotImplementedError()

    def create_continuous(self) -> tuple[_Discrete, _Continuous, _Continuous]:
        return self.t_arr, self.u_t, self.y_t

    @property
    @abstractmethod
    def u_t(self) -> _Continuous:
        raise NotImplementedError()

    @property
    @abstractmethod
    def y_t(self) -> _Continuous:
        raise NotImplementedError()

    def create_discrete(self) -> tuple[_Discrete, _Discrete, _Discrete]:
        return self.t_arr, self.u_arr, self.y_arr


class AbstractContinuousDataset(AbstractDataset, ABC):
    @property
    def u_arr(self) -> _Discrete:
        return self.u_t(self.t_arr)

    @property
    def y_arr(self) -> _Discrete:
        return self.y_t(self.t_arr)


class AbstractDiscreteDataset(AbstractDataset, ABC):
    @property
    def u_t(self) -> _Continuous:
        def u_t(t: _Discrete) -> _Discrete:
            idx = np.floor(np.asarray(t) / self._t_step + 1e-15).astype(int)
            return self.u_arr[idx]

        return u_t

    @property
    def y_t(self) -> _Continuous:
        def y_t(t: _Discrete) -> _Discrete:
            idx = np.floor(np.asarray(t) / self._t_step + 1e-15).astype(int)
            return self.y_arr[idx]

        return y_t


class DelayedSineDataset(AbstractContinuousDataset):  # 本質的に連続的なデータセット
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

    def __f_t(self, t: _Discrete) -> _Discrete:
        return self._amplitude * np.sin(2 * np.pi * self._freq * t + self._phase_offset)

    @property
    def u_t(self) -> _Continuous:
        def u_t(t: _Discrete) -> _Discrete:
            return self.__f_t(t)

        return u_t

    @property
    def y_t(self) -> _Continuous:
        def y_t(t: _Discrete) -> _Discrete:
            return self.__f_t(t - self._discrete_lag * self._t_step)

        return y_t


class NoisyDelayedSineDataset(DelayedSineDataset):
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
    def u_t(self) -> _Continuous:  # 生成の度に異なるノイズが加わる
        super_u_t = super().u_t

        def u_t(t: _Discrete) -> _Discrete:
            a = super_u_t(t)
            return a + self._rng.normal(0, self._noise_std, len(a))

        return u_t

    @property
    def y_t(self) -> _Continuous:  # 生成の度に異なるノイズが加わる
        super_y_t = super().y_t

        def y_t(t: _Discrete) -> _Discrete:
            a = super_y_t(t)
            return a + self._rng.normal(0, self._noise_std, len(a))

        return y_t


class DelayedRandomDataset(AbstractDiscreteDataset):  # 本質的に離散的なデータセット
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
    def u_arr(self) -> _Discrete:
        return self._series

    @property
    def y_arr(self) -> _Discrete:
        y_arr = np.zeros_like(self.u_arr)
        y_arr[self._discrete_lag:] = self.u_arr[:-self._discrete_lag]
        return y_arr


class ParityCheckDataset(AbstractDiscreteDataset):  # 本質的に離散的なデータセット
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
    def u_arr(self) -> _Discrete:
        return self._series

    @property
    def y_arr(self) -> _Discrete:
        y_arr = np.zeros_like(self.u_arr)
        for k in range(self._window_size, len(self.u_arr)):
            y_arr[k] = np.sum(self.u_arr[k - self._window_size:k]) % 2
        return y_arr


def _preview_dataset():
    t_max = 5
    t_step = 0.1
    discrete_lag = 3
    noise_std = 0.05

    ds_delayed_sine = DelayedSineDataset(
        t_max=t_max,
        t_step=t_step,
        freq=1,
        phase_offset=np.pi / 2,
        discrete_lag=discrete_lag,
        amplitude=1,
    )

    ds_noisy_delayed_sine = NoisyDelayedSineDataset(
        t_max=t_max,
        t_step=t_step,
        freq=1,
        phase_offset=np.pi / 2,
        discrete_lag=discrete_lag,
        amplitude=1,
        noise_std=noise_std,
    )

    ds_delayed_random = DelayedRandomDataset(
        t_max=t_max,
        t_step=t_step,
        discrete_lag=discrete_lag,
        low=0,
        high=1,
    )

    ds_parity_check = ParityCheckDataset(
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
