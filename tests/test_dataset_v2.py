import itertools
from typing import Any

import numpy as np
import pytest

from utils.dataset_v2 import AbstractDiscreteDatasetGenerator, DelayedRandomDatasetGenerator, \
    DelayedSineDatasetGenerator, \
    ParityCheckDatasetGenerator, NoisyDelayedSineDatasetGenerator, NarmaDatasetGenerator


# TODO: u_t, y_t: Continuousにスカラー値を入れた時のテストを追加

class _TestDiscreteDataset(AbstractDiscreteDatasetGenerator):
    def __init__(self, *, t_max: float, t_step: float):
        super().__init__(t_max=t_max, t_step=t_step)

    @property
    def name(self) -> str:
        return "TestDiscreteDataset"

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(
            t_max=self._t_max,
            t_step=self._t_step,
        )

    @property
    def u_arr(self) -> np.ndarray:
        return self.t_arr

    @property
    def y_arr(self) -> np.ndarray:
        return self.t_arr


@pytest.mark.parametrize(
    "t_max,t_step",
    [
        (t_max, t_step)
        for t_max, t_step in itertools.product(
        [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000],
        [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1],
    )
        if t_max > t_step and t_max / t_step < 100000
    ]
)
def test_abstract_discrete_dataset(t_max, t_step):
    ds = _TestDiscreteDataset(t_max=t_max, t_step=t_step)

    t_arr_ref = np.arange(0, t_max, t_step)
    np.testing.assert_array_equal(ds.t_arr, t_arr_ref)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 2), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 3), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 4), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 5), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 6), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 7), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 8), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr + t_step / 10 * 9), ds.u_arr)
    np.testing.assert_array_equal(ds.u_t(ds.t_arr[:-1] + t_step), ds.u_arr[1:])


def test_delayed_sine_dataset():
    # DelayedSineDatasetの基本的な動作（離散・連続データの一致）を検証するテスト
    t_max = 1
    t_step = 0.01
    amplitude = 1
    freq = 1
    phase_offset = np.pi / 2
    discrete_lag = 3
    ds = DelayedSineDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        freq=freq,
        phase_offset=phase_offset,
        discrete_lag=discrete_lag,
        amplitude=amplitude,
    )

    # 離散時刻データ
    t_arr_ref = np.arange(0, t_max, t_step)
    np.testing.assert_array_almost_equal(ds.t_arr, t_arr_ref)

    # 離散入力データ
    u_arr_ref = amplitude * np.sin(2 * np.pi * freq * ds.t_arr + phase_offset)
    np.testing.assert_array_almost_equal(ds.u_arr, u_arr_ref)

    # 離散出力データ
    y_arr_ref = amplitude * np.sin(
        2 * np.pi * freq * (ds.t_arr - discrete_lag * t_step) + phase_offset)
    np.testing.assert_array_almost_equal(ds.y_arr, y_arr_ref)

    # 離散出力データと離散入力データの関係
    np.testing.assert_array_almost_equal(ds.y_arr[discrete_lag:], ds.u_arr[:-discrete_lag])

    # 連続入力データのサンプリング
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_arr)

    # 連続出力データのサンプリング
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_arr)


# noinspection DuplicatedCode
def test_delayed_sine_dataset_uniqueness():
    # DelayedSineDatasetで同一インスタンスの1回目生成されたデータと2回目に生成されたデータが同一かどうかを確認するテスト
    ds = DelayedSineDatasetGenerator(
        t_max=1,
        t_step=0.1,
        freq=1,
        phase_offset=np.pi / 2,
        discrete_lag=3,
        amplitude=1,
    )

    np.testing.assert_array_almost_equal(ds.t_arr, ds.t_arr)
    np.testing.assert_array_almost_equal(ds.u_arr, ds.u_arr)
    np.testing.assert_array_almost_equal(ds.y_arr, ds.y_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_t(ds.t_arr))
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_t(ds.t_arr))


def test_noisy_delayed_sine_dataset_noise_behavior():
    # NoisyDelayedSineDatasetのノイズ分散が理論値に近いことを統計的に検証するテスト
    t_max = 1
    t_step = 0.1
    freq = 1
    phase_offset = np.pi / 2
    discrete_lag = 3
    amplitude = 2
    noise_std = 0.01
    n_repeat = 2000

    ds_pure = DelayedSineDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        freq=freq,
        phase_offset=phase_offset,
        discrete_lag=discrete_lag,
        amplitude=amplitude,
    )

    ds = NoisyDelayedSineDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        freq=freq,
        phase_offset=phase_offset,
        discrete_lag=discrete_lag,
        amplitude=amplitude,
        noise_std=noise_std,
        seed=42,
    )

    # sample data
    u_arrs = []
    y_arrs = []
    for _ in range(n_repeat):
        u_arrs.append(ds.u_arr)
        y_arrs.append(ds.y_arr)
    u_arrs = np.stack(u_arrs)
    y_arrs = np.stack(y_arrs)

    # compare pure and noisy generator
    u_arr_mean = u_arrs.mean(axis=0)
    y_arr_mean = y_arrs.mean(axis=0)
    np.testing.assert_allclose(u_arr_mean, ds_pure.u_arr, rtol=noise_std / n_repeat ** .5 * 5)
    np.testing.assert_allclose(y_arr_mean, ds_pure.y_arr, rtol=noise_std / n_repeat ** .5 * 5)

    # evaluate noise behavior
    u_std = u_arrs.std(axis=0)
    y_std = y_arrs.std(axis=0)
    np.testing.assert_allclose(u_std, noise_std, rtol=0.1)
    np.testing.assert_allclose(y_std, noise_std, rtol=0.1)


def test_delayed_random_dataset():
    # DelayedRandomDatasetの基本的な動作（乱数系列・遅延出力）を検証するテスト
    t_max = 1
    t_step = 0.01
    discrete_lag = 3
    low = 0
    high = 1
    ds = DelayedRandomDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        discrete_lag=discrete_lag,
        low=low,
        high=high,
        seed=42,
    )

    rng = np.random.RandomState(42)

    # 離散時刻データ
    t_arr_ref = np.arange(0, t_max, t_step)
    np.testing.assert_array_almost_equal(ds.t_arr, t_arr_ref)

    # 乱数データ
    data_series = rng.uniform(low, high, len(ds.t_arr))

    # 離散入力データ
    np.testing.assert_array_almost_equal(ds.u_arr, data_series)

    # 離散出力データ
    y_arr_ref = np.concatenate([[0, 0, 0], data_series[:-discrete_lag]])
    np.testing.assert_array_almost_equal(ds.y_arr, y_arr_ref)

    # 離散出力データと離散入力データの関係
    np.testing.assert_array_almost_equal(ds.y_arr[discrete_lag:], ds.u_arr[:-discrete_lag])

    # 連続入力データのサンプリング
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr + (t_step - 1e-10)), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr[:-1] + t_step), ds.u_arr[1:])

    # 連続出力データのサンプリング
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr + (t_step - 1e-10)), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr[:-1] + t_step), ds.y_arr[1:])


def test_delayed_random_dataset_uniqueness():
    # DelayedRandomDatasetで同一インスタンスの1回目生成されたデータと2回目に生成されたデータが同一かどうかを確認するテスト
    t_max = 1
    t_step = 0.1
    discrete_lag = 3
    low = 0
    high = 1
    seed = 42

    ds = DelayedRandomDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        discrete_lag=discrete_lag,
        low=low,
        high=high,
        seed=seed,
    )

    np.testing.assert_array_almost_equal(ds.t_arr, ds.t_arr)
    np.testing.assert_array_almost_equal(ds.u_arr, ds.u_arr)
    np.testing.assert_array_almost_equal(ds.y_arr, ds.y_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_t(ds.t_arr))
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_t(ds.t_arr))


def test_parity_check_dataset():
    # ParityCheckDatasetの動作（パリティ計算・連続/離散データの一致）を検証するテスト
    t_max = 1
    t_step = 0.1
    ds = ParityCheckDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        window_size=3,
        seed=42,
    )

    rng = np.random.RandomState(42)

    # 離散時刻データ
    t_arr_ref = np.arange(0, t_max, t_step)
    np.testing.assert_array_almost_equal(ds.t_arr, t_arr_ref)

    # 離散入力データ
    expected_u_arr = rng.choice([0, 1], size=len(ds.t_arr))
    np.testing.assert_array_almost_equal(ds.u_arr, expected_u_arr)

    # 離散出力データ
    for i in range(3, len(ds.t_arr)):
        assert ds.y_arr[i] == (ds.u_arr[i - 3] + ds.u_arr[i - 2] + ds.u_arr[i - 1]) % 2

    # 連続入力データのサンプリング
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr + (t_step - 1e-10)), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr[:-1] + t_step), ds.u_arr[1:])

    # 連続出力データのサンプリング
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr + (t_step - 1e-10)), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr[:-1] + t_step), ds.y_arr[1:])


def test_parity_check_dataset_uniqueness():
    # ParityCheckDatasetで同一インスタンスの1回目生成されたデータと2回目に生成されたデータが同一かどうかを確認するテスト
    ds = ParityCheckDatasetGenerator(
        t_max=1,
        t_step=0.1,
        window_size=3,
        seed=42,
    )

    np.testing.assert_array_almost_equal(ds.t_arr, ds.t_arr)
    np.testing.assert_array_almost_equal(ds.u_arr, ds.u_arr)
    np.testing.assert_array_almost_equal(ds.y_arr, ds.y_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_t(ds.t_arr))
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_t(ds.t_arr))


def _create_narma_reference(u_arr, n=10, a=0.3, b=0.05, c=1.5, d=0.1):
    y_arr = np.zeros_like(u_arr)
    for k in range(n - 1, len(u_arr) - 1):
        y_arr[k + 1] = (
                a * y_arr[k]
                + b * y_arr[k] * np.sum(y_arr[k - n + 1:k + 1])
                + c * u_arr[k] * u_arr[k - n + 1]
                + d
        )
    return y_arr


def test_narma_dataset():
    # NarmaDatasetの基本動作（乱数系列・NARMA出力・サンプリング）を検証するテスト
    t_max = 1
    t_step = 0.1
    n = 10
    a, b, c, d = 0.3, 0.05, 1.5, 0.1
    low, high = 0.0, 0.5
    seed = 42

    ds = NarmaDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        n=n,
        a=a,
        b=b,
        c=c,
        d=d,
        low=low,
        high=high,
        seed=seed,
    )

    rng = np.random.RandomState(seed)
    u_arr_ref = rng.uniform(low, high, len(ds.t_arr))
    y_arr_ref = _create_narma_reference(u_arr_ref, n=n, a=a, b=b, c=c, d=d)

    # 離散時刻データ
    np.testing.assert_array_almost_equal(ds.t_arr, np.arange(0, t_max, t_step))

    # 離散入力データ
    np.testing.assert_array_almost_equal(ds.u_arr, u_arr_ref)

    # 離散出力データ
    np.testing.assert_array_almost_equal(ds.y_arr, y_arr_ref)

    # 連続入力データのサンプリング
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr + (t_step - 1e-10)), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr[:-1] + t_step), ds.u_arr[1:])

    # 連続出力データのサンプリング
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr + (t_step - 1e-10)), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr[:-1] + t_step), ds.y_arr[1:])


def test_narma_dataset_uniqueness():
    # NarmaDatasetで同一インスタンスの1回目生成されたデータと2回目に生成されたデータが同一かどうかを確認するテスト
    t_max = 1
    t_step = 0.01
    n = 10
    a, b, c, d = 0.3, 0.05, 1.5, 0.1
    low, high = 0.0, 0.5
    seed = 42

    ds = NarmaDatasetGenerator(
        t_max=t_max,
        t_step=t_step,
        n=n,
        a=a,
        b=b,
        c=c,
        d=d,
        low=low,
        high=high,
        seed=seed,
    )

    # 1回目と2回目のu_arr, y_arr, u_t, y_tが同じであることを確認
    np.testing.assert_array_almost_equal(ds.u_arr, ds.u_arr)
    np.testing.assert_array_almost_equal(ds.y_arr, ds.y_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_t(ds.t_arr))
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_t(ds.t_arr))
