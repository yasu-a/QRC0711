import numpy as np
import pytest

from utils.dataset import Narma, to_continuous_function, DelayedSine
from utils.dataset import train_test_split


def _create_narma_10(u_arr):
    y_arr = np.zeros_like(u_arr)
    for t in range(10, len(u_arr)):
        y_arr[t] = 0.3 * y_arr[t - 1] \
                   + 0.05 * y_arr[t - 1] * np.sum(y_arr[t - 10:t]) \
                   + 1.5 * u_arr[t - 10] * u_arr[t - 1] \
                   + 0.1
    return y_arr


def test_random_input_generation():
    n = 1000
    low = 0.0
    high = 0.5
    u_arr_ref = np.random.RandomState(0).uniform(low, high, n)
    assert u_arr_ref.shape == (n,)

    u_arr = Narma.create_random_input(n=n)
    assert u_arr.shape == (n,)

    np.testing.assert_array_almost_equal(u_arr_ref, u_arr, decimal=10)


def test_narma10_default_parameters():
    """デフォルトパラメータでのNarma10ラスの動作をテスト"""
    n = 1000
    u_arr, y_arr = Narma.create_with_random_input(n=n)
    assert u_arr.shape == (n,)
    assert y_arr.shape == (n,)

    # _create_narma_10関数で計算
    y_arr_reference = _create_narma_10(u_arr)
    assert y_arr_reference.shape == (n,)

    # 結果を比較
    np.testing.assert_array_almost_equal(y_arr, y_arr_reference, decimal=10)


@pytest.mark.parametrize("case", [
    {
        # No washout, expect normal split
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": None,
        "n_test_washout": None,
        "n_washout": None,
        "is_error_expected": False,
        "expected_train": np.arange(0, 21),
        "expected_test": np.arange(21, 30),
    },
    {
        # Only n_train_washout specified
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": 20,
        "n_test_washout": None,
        "n_washout": None,
        "is_error_expected": False,
        "expected_train": np.arange(20, 27),
        "expected_test": np.arange(27, 30),
    },
    {
        # Only n_test_washout specified
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": None,
        "n_test_washout": 20,
        "n_washout": None,
        "is_error_expected": False,
        "expected_train": np.arange(0, 7),
        "expected_test": np.arange(27, 30),
    },
    {
        # Both n_train_washout and n_test_washout specified
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": 15,
        "n_test_washout": 5,
        "n_washout": None,
        "is_error_expected": False,
        "expected_train": np.arange(15, 22),
        "expected_test": np.arange(27, 30),
    },
    {
        # n_washout and n_train_washout specified (should raise error)
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": 20,
        "n_test_washout": None,
        "n_washout": 20,
        "is_error_expected": True,
        "expected_train": None,
        "expected_test": None,
    },
    {
        # n_washout and n_test_washout specified (should raise error)
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": None,
        "n_test_washout": 20,
        "n_washout": 20,
        "is_error_expected": True,
        "expected_train": None,
        "expected_test": None,
    },
    {
        # n_washout and both n_train_washout/n_test_washout specified (should raise error)
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": 15,
        "n_test_washout": 5,
        "n_washout": 20,
        "is_error_expected": True,
        "expected_train": None,
        "expected_test": None,
    },
    {
        # Only n_washout specified
        "n_total": 30,
        "test_ratio": 0.3,
        "n_train_washout": None,
        "n_test_washout": None,
        "n_washout": 10,
        "is_error_expected": False,
        "expected_train": np.arange(10, 17),
        "expected_test": np.arange(27, 30),
    },
])
def test_train_test_split_with_separated_washout(case):
    u_arr = np.arange(case["n_total"])
    y_arr = np.arange(case["n_total"]) + 100
    kwargs = {}
    if case["n_train_washout"] is not None:
        kwargs["n_train_washout"] = case["n_train_washout"]
    if case["n_test_washout"] is not None:
        kwargs["n_test_washout"] = case["n_test_washout"]
    if case["n_washout"] is not None:
        kwargs["n_washout"] = case["n_washout"]

    if case["is_error_expected"]:
        with pytest.raises(ValueError):
            train_test_split(u_arr, y_arr, test_ratio=case["test_ratio"], **kwargs)
    else:
        (u_train, u_test), (y_train, y_test) \
            = train_test_split(u_arr, y_arr, test_ratio=case["test_ratio"], **kwargs)
        np.testing.assert_array_equal(u_train, case["expected_train"])
        np.testing.assert_array_equal(y_train, case["expected_train"] + 100)
        np.testing.assert_array_equal(u_test, case["expected_test"])
        np.testing.assert_array_equal(y_test, case["expected_test"] + 100)


def test_to_continuous_function():
    a = np.arange(5)
    f = to_continuous_function(a, delta_t=0.1)
    f = np.vectorize(f)
    t_test = np.array([0, 0.05, 0.099999999, 0.1, 0.499999999])
    y_test = np.array([0, 0, 0, 1, 4])
    np.testing.assert_array_equal(f(t_test), y_test)


@pytest.mark.parametrize("n", [1, 5, 20])
def test_delayed_sin(n):
    t_arr = np.linspace(0, 10, 100)
    u_arr, y_arr = DelayedSine.create(t_arr, noise_std=0, seed=0, lag=n)
    np.testing.assert_array_equal(u_arr[:-n], y_arr[n:])
