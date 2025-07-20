import concurrent.futures
import itertools
from typing import Generic

from tqdm import tqdm

from search.base import AbstractParameterSearcher, ParamType


class GridParameterSearcher(AbstractParameterSearcher, Generic[ParamType]):
    def _list_param_objects(self) -> list[ParamType]:
        # パラメータの組み合わせを生成
        return [
            self._param_mapper(dict(zip(self._param_grid.keys(), values)))
            for values in itertools.product(*self._param_grid.values())
        ]

    def _run_search(self, *, n_workers: int) -> None:
        param_objects = self._list_param_objects()

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self._eval_score, p): p for p in param_objects}
            bar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))
            for future in bar:
                param = futures[future]
                score = future.result()
                self._history.append((param, score))
                bar.set_description(f"Best score={max(score for _, score in self._history):.3f}")

        if not self._history:
            raise RuntimeError("No valid parameter found")
