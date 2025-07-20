import copy
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Collection, Any

ParamType = TypeVar("ParamType")


class AbstractParameterSearcher(ABC, Generic[ParamType]):
    def __init__(
            self,
            scorer: Callable[[ParamType], float],
            param_grid: dict[str, Collection[Any]],  # Collection[Any]からlist[Any]に変更
            param_mapper: Callable[[dict[str, Any]], ParamType],
            forbid: list[dict[str, Any]] | None = None,
    ):
        self.__scorer = scorer
        self._param_grid = param_grid
        self._param_mapper = param_mapper
        self._forbid = forbid or []

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

    def _eval_score(self, p: ParamType) -> float:
        score = self.__scorer(p)
        assert isinstance(score, float), (type(score), score)
        return score

    def _is_forbidden(self, param_dict: dict[str, Any]) -> bool:
        """パラメータが禁止領域に含まれるかチェック"""
        for forbidden_config in self._forbid:
            if all(param_dict[key] == forbidden_value for key, forbidden_value in forbidden_config.items()):
                return True
        return False

    @abstractmethod
    def _run_search(self, *, n_workers: int) -> None:
        raise NotImplementedError()

    def search(self, *, n_workers: int) -> tuple[ParamType, float]:  # best param and score
        self._run_search(n_workers=n_workers)
        return self.best_param, self.best_score
