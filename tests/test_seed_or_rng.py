import numpy as np
import pytest

from core.seed_or_rng import check_seed_or_rng_and_get_rng


def test_check_seed_or_rng_and_get_rng_default():
    # 両方未指定: デフォルトseed=0
    rng_get = check_seed_or_rng_and_get_rng()
    rng_true = np.random.RandomState(0)
    arr_get = rng_get.random(10)
    arr_true = rng_true.random(10)
    np.testing.assert_array_almost_equal(arr_get, arr_true)


def test_check_seed_or_rng_and_get_rng_seed_only():
    # seedのみ指定
    rng_get = check_seed_or_rng_and_get_rng(seed=123)
    rng_true = np.random.RandomState(123)
    arr_get = rng_get.random(10)
    arr_true = rng_true.random(10)
    np.testing.assert_array_almost_equal(arr_get, arr_true)


def test_check_seed_or_rng_and_get_rng_rng_only():
    # rngのみ指定
    rng_get = check_seed_or_rng_and_get_rng(rng=np.random.RandomState(456))
    rng_true = np.random.RandomState(456)
    arr_get = rng_get.random(10)
    arr_true = rng_true.random(10)
    np.testing.assert_array_almost_equal(arr_get, arr_true)


def test_check_seed_or_rng_and_get_rng_both_error():
    # 両方指定はエラー
    with pytest.raises(ValueError):
        check_seed_or_rng_and_get_rng(seed=1, rng=np.random.RandomState(2))
