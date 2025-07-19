import concurrent.futures
import copy
import itertools
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Collection, Any

from tqdm import tqdm

ParamType = TypeVar("ParamType")


class AbstractParameterSearcher(ABC, Generic[ParamType]):
    def __init__(
            self,
            scorer: Callable[[ParamType], float],
            param_grid: dict[str, Collection[Any]],
            param_mapper: Callable[[dict[str, Any]], ParamType],
    ):
        self._scorer = scorer
        self._param_grid = param_grid
        self._param_mapper = param_mapper

        self._history: list[tuple[ParamType, float]] = []  # list of parameters and scores

    @property
    def best_score(self) -> float:
        if not self._history:
            raise ValueError("no search history recorded")
        return max(score for _, score in self._history)

    @property
    def best_param(self) -> ParamType:
        if not self._history:
            raise ValueError("no search history recorded")
        return max(self._history, key=lambda x: x[1])[0]

    @property
    def history(self) -> list[tuple[ParamType, float]]:
        return copy.deepcopy(self._history)

    @abstractmethod
    def _run_search(self, *, n_workers: int) -> None:
        raise NotImplementedError()

    def search(self, *, n_workers: int) -> tuple[ParamType, float]:  # best param and score
        self._run_search(n_workers=n_workers)
        return self.best_param, self.best_score


class GridParameterSearcher(AbstractParameterSearcher, Generic[ParamType]):
    def _list_param_objects(self) -> list[ParamType]:
        # パラメータの組み合わせを生成
        return [
            self._param_mapper(dict(zip(self._param_grid.keys(), values)))
            for values in itertools.product(*self._param_grid.values())
        ]

    def _run_search(self, *, n_workers: int) -> tuple[ParamType, float]:
        best_score = -float('inf')
        best_param = None
        param_objects = self._list_param_objects()

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self._scorer, p): p for p in param_objects}
            bar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))
            for future in bar:
                param = futures[future]
                score = future.result()
                if score > best_score:
                    best_score = score
                    best_param = param
                bar.set_description(f"Best score={best_score:.3f}")

        if best_param is None:
            raise RuntimeError("No valid parameter found")

        return best_param, best_score
