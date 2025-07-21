import numpy as np

from utils.dataset_v2 import DelayedRandomDataset, DelayedSineDataset, ParityCheckDataset, \
    NoisyDelayedSineDataset


def test_delayed_sine_dataset():
    # DelayedSineDatasetの基本的な動作（離散・連続データの一致）を検証するテスト
    ds = DelayedSineDataset(
        t_max=1,
        t_step=0.1,
        freq=1,
        phase_offset=np.pi / 2,
        discrete_lag=3,
        amplitude=1,
    )

    # 離散時刻データ
    t_arr_expected = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    np.testing.assert_array_almost_equal(ds.t_arr, t_arr_expected)

    # 離散入力データ
    np.testing.assert_array_almost_equal(ds.u_arr, np.sin(2 * np.pi * 1 * ds.t_arr + np.pi / 2))

    # 離散出力データ
    y_arr_expected = np.sin(2 * np.pi * 1 * (ds.t_arr - 3 * 0.1) + np.pi / 2)
    np.testing.assert_array_almost_equal(ds.y_arr, y_arr_expected)

    # 離散出力データと離散入力データの関係
    np.testing.assert_array_almost_equal(ds.y_arr[3:], ds.u_arr[:-3])

    # 連続入力データのサンプリング
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_arr)

    # 連続出力データのサンプリング
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_arr)


# noinspection DuplicatedCode
def test_delayed_sine_dataset_uniqueness():
    # DelayedSineDatasetで同一インスタンスの1回目生成されたデータと2回目に生成されたデータが同一かどうかを確認するテスト
    ds = DelayedSineDataset(
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

    ds = NoisyDelayedSineDataset(
        t_max=t_max,
        t_step=t_step,
        freq=freq,
        phase_offset=phase_offset,
        discrete_lag=discrete_lag,
        amplitude=amplitude,
        noise_std=noise_std,
        seed=42,
    )

    u_arrs = []
    y_arrs = []
    for _ in range(n_repeat):
        u_arrs.append(ds.u_arr)
        y_arrs.append(ds.y_arr)
    u_arrs = np.stack(u_arrs)
    y_arrs = np.stack(y_arrs)

    u_std = u_arrs.std(axis=0)
    y_std = y_arrs.std(axis=0)

    np.testing.assert_allclose(u_std, noise_std, rtol=0.1)
    np.testing.assert_allclose(y_std, noise_std, rtol=0.1)


def test_delayed_random_dataset():
    # DelayedRandomDatasetの基本的な動作（乱数系列・遅延出力）を検証するテスト
    ds = DelayedRandomDataset(
        t_max=1,
        t_step=0.1,
        discrete_lag=3,
        low=0,
        high=1,
        seed=42,
    )

    rng = np.random.RandomState(42)

    # 離散時刻データ
    t_arr_expected = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    np.testing.assert_array_almost_equal(ds.t_arr, t_arr_expected)

    # 乱数データ
    data_series = rng.uniform(0, 1, len(ds.t_arr))

    # 離散入力データ
    np.testing.assert_array_almost_equal(ds.u_arr, data_series)

    # 離散出力データ
    y_arr_expected = np.concatenate([[0, 0, 0], data_series[:-3]])
    np.testing.assert_array_almost_equal(ds.y_arr, y_arr_expected)

    # 離散出力データと離散入力データの関係
    np.testing.assert_array_almost_equal(ds.y_arr[3:], ds.u_arr[:-3])

    # 連続入力データのサンプリング
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr + (0.1 - 1e-10)), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr[:-1] + 0.1), ds.u_arr[1:])

    # 連続出力データのサンプリング
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr + (0.1 - 1e-10)), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr[:-1] + 0.1), ds.y_arr[1:])


def test_delayed_random_dataset_uniqueness():
    # DelayedRandomDatasetで同一インスタンスの1回目生成されたデータと2回目に生成されたデータが同一かどうかを確認するテスト
    ds = DelayedRandomDataset(
        t_max=1,
        t_step=0.1,
        discrete_lag=3,
        low=0,
        high=1,
        seed=42,
    )

    np.testing.assert_array_almost_equal(ds.t_arr, ds.t_arr)
    np.testing.assert_array_almost_equal(ds.u_arr, ds.u_arr)
    np.testing.assert_array_almost_equal(ds.y_arr, ds.y_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_t(ds.t_arr))
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_t(ds.t_arr))


def test_parity_check_dataset():
    # ParityCheckDatasetの動作（パリティ計算・連続/離散データの一致）を検証するテスト
    ds = ParityCheckDataset(
        t_max=1,
        t_step=0.1,
        window_size=3,
        seed=42,
    )

    rng = np.random.RandomState(42)

    # 離散時刻データ
    t_arr_expected = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    np.testing.assert_array_almost_equal(ds.t_arr, t_arr_expected)

    # 離散入力データ
    expected_u_arr = rng.choice([0, 1], size=len(ds.t_arr))
    np.testing.assert_array_almost_equal(ds.u_arr, expected_u_arr)

    # 離散出力データ
    for i in range(3, len(ds.t_arr)):
        assert ds.y_arr[i] == (ds.u_arr[i - 3] + ds.u_arr[i - 2] + ds.u_arr[i - 1]) % 2

    # 連続入力データのサンプリング
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr + (0.1 - 1e-10)), ds.u_arr)
    np.testing.assert_array_almost_equal(ds.u_t(ds.t_arr[:-1] + 0.1), ds.u_arr[1:])

    # 連続出力データのサンプリング
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr + (0.1 - 1e-10)), ds.y_arr)
    np.testing.assert_array_almost_equal(ds.y_t(ds.t_arr[:-1] + 0.1), ds.y_arr[1:])


def test_parity_check_dataset_uniqueness():
    # ParityCheckDatasetで同一インスタンスの1回目生成されたデータと2回目に生成されたデータが同一かどうかを確認するテスト
    ds = ParityCheckDataset(
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
