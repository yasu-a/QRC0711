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


def param_mapper(x):
    return TestParamType(**x)


def scorer(x):
    return x.a + x.b + x.c + x.d + len(x.e)


param_grid: dict[str, list[Any]] = dict(
    a=[1, 2, 3, 4],
    b=[10, 20, 30, 40],
    c=[100, 200, 300, 400],
    d=[0.1, 0.2, 0.3, 0.4],
    e=["a", "aa", "aaa", "aaaa"],
)

forbid = [
    dict(e="aa"),
    dict(a=3, b=30, c=300, d=0.3, e="aaa"),
    dict(b=40, c=400),
]


def test_grid_search_with_forbid():
    search = GridParameterSearcher[TestParamType](
        param_grid=param_grid,  # type: ignore
        param_mapper=param_mapper,
        scorer=scorer,
        forbid=forbid,
    )
    search.search(n_workers=2)
    assert search.best_param == TestParamType(a=4, b=30, c=400, d=0.4, e="aaaa")
    assert search.best_score == 438.4


def test_ga_search_with_forbid():
    search = GAParameterSearcher[TestParamType](
        param_grid=param_grid,  # type: ignore
        param_mapper=param_mapper,
        scorer=scorer,
        forbid=forbid,
        n_pop=20,
        n_gen=5,
        mutation_rate=0.1,
        crossover_rate=0.8,
        tournament_size=5,
        seed=1,
    )
    search.search(n_workers=2)
    assert search.best_score == 438.4
    assert search.best_param == TestParamType(a=4, b=30, c=400, d=0.4, e="aaaa")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df = pd.DataFrame([
            {
                **asdict(param),
                "_score": score,
            } for param, score in search.history
        ])
        df = df.sort_values(by="_score", ascending=False)
        print(df)
