from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import pytest

from search.ga import GAParameterSearcher
from search.grid import GridParameterSearcher


@dataclass
class TestParamType:
    a: int
    b: int
    c: int
    d: float
    e: str


def display_history(search):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df = pd.DataFrame([
            {
                **asdict(param),
                "_score": score,
            } for param, score in search.history
        ])
        df = df.sort_values(by="_score", ascending=False)
        print("Head 10:")
        print(df.head(10))
        print("Tail 10:")
        print(df.tail(10))


def _param_mapper(x):
    return TestParamType(**x)


def _param_scorer(x):
    return x.a + x.b + x.c + x.d + len(x.e)


_param_grid: dict[str, list[Any]] = dict(
    a=[1, 2, 3, 4],
    b=[10, 20, 30, 40],
    c=[100, 200, 300, 400],
    d=[0.1, 0.2, 0.3, 0.4],
    e=["a", "aa", "aaa", "aaaa"],
)

_expected_best_param_no_constraint = TestParamType(a=4, b=40, c=400, d=0.4, e="aaaa")
_expected_best_score_no_constraint = 448.4


@pytest.mark.parametrize("n_workers", [1, 2])
def test_grid_search_no_constraint(n_workers):
    search = GridParameterSearcher[TestParamType](
        param_grid=_param_grid,  # type: ignore
        param_mapper=_param_mapper,
        scorer=_param_scorer,
    )
    search.search(n_workers=n_workers)
    assert search.best_score == _expected_best_score_no_constraint
    assert search.best_param == _expected_best_param_no_constraint
    display_history(search)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_ga_search_no_constraint(n_workers):
    search = GAParameterSearcher[TestParamType](
        param_grid=_param_grid,  # type: ignore
        param_mapper=_param_mapper,
        scorer=_param_scorer,
        n_pop=30,
        n_gen=4,
        mutation_rate=0.01,
        crossover_rate=0.9,
        crossover_type="uniform",
        tournament_size=3,
        seed=2,
    )
    search.search(n_workers=n_workers)
    assert search.best_score == _expected_best_score_no_constraint
    assert search.best_param == _expected_best_param_no_constraint
    display_history(search)


def test_ga_search_no_constraint_two_point_crossover():
    search = GAParameterSearcher[TestParamType](
        param_grid=_param_grid,  # type: ignore
        param_mapper=_param_mapper,
        scorer=_param_scorer,
        n_pop=30,
        n_gen=20,  # テスト用の問題に対して2点交叉は効率が悪いので多めに設定
        mutation_rate=0.01,
        crossover_rate=0.9,
        crossover_type="two-point",
        tournament_size=3,
        seed=2,
    )
    search.search(n_workers=1)
    assert search.best_score == _expected_best_score_no_constraint
    assert search.best_param == _expected_best_param_no_constraint
    display_history(search)


def _constraint_predicate_fn(p: TestParamType) -> bool:
    if p.e == "aa":
        return False
    if (p.a == 3
            and p.b == 30
            and p.c == 300
            and p.d == 0.3
            and p.e == "aaa"):
        return False
    if p.b == 40 and p.c == 400:
        return False
    return True


_expected_best_param_with_constraint = TestParamType(a=4, b=30, c=400, d=0.4, e="aaaa")
_expected_best_score_with_constraint = 438.4


@pytest.mark.parametrize("n_workers", [1, 2])
def test_grid_search_with_constraint(n_workers):
    search = GridParameterSearcher[TestParamType](
        param_grid=_param_grid,  # type: ignore
        param_mapper=_param_mapper,
        scorer=_param_scorer,
        constraint_predicate=_constraint_predicate_fn,
    )
    search.search(n_workers=n_workers)
    assert search.best_score == _expected_best_score_with_constraint
    assert search.best_param == _expected_best_param_with_constraint
    display_history(search)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_ga_search_with_constraint(n_workers):
    search = GAParameterSearcher[TestParamType](
        param_grid=_param_grid,  # type: ignore
        param_mapper=_param_mapper,
        scorer=_param_scorer,
        constraint_predicate=_constraint_predicate_fn,
        n_pop=30,
        n_gen=4,
        mutation_rate=0.01,
        crossover_rate=0.9,
        crossover_type="uniform",
        tournament_size=3,
        seed=2,
    )
    search.search(n_workers=n_workers)
    assert search.best_score == _expected_best_score_with_constraint
    assert search.best_param == _expected_best_param_with_constraint
    display_history(search)
