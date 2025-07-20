import concurrent.futures
import itertools
from typing import Generic, Iterable

from tqdm import tqdm

from search.base import AbstractParameterSearcher, ParamType


class GridParameterSearcher(AbstractParameterSearcher, Generic[ParamType]):
    _SPACE_LIMIT = 1e+6

    def _list_param_objects(self) -> list[ParamType]:
        """
        Generate all feasible parameter combinations.

        Returns:
            A list of feasible parameter objects.

        Raises:
            RuntimeError: If the number of parameter combinations exceeds the limit.
        """
        param_lst: list[ParamType] = []
        # Generate all combinations of parameter values
        for values in itertools.product(*self._param_grid.values()):
            dct = dict(zip(self._param_grid.keys(), values))
            param: ParamType = self._param_mapper(dct)
            if self._is_feasible(param):
                param_lst.append(param)
            if len(param_lst) >= self._SPACE_LIMIT:
                raise RuntimeError("The number of parameters exceeds the limit")
        return param_lst

    def _eval_parameters_single(self, param_lst: list[ParamType]) \
            -> Iterable[tuple[ParamType, float]]:
        """
        Evaluate all parameters sequentially.

        Args:
            param_lst: List of parameter objects to evaluate.

        Yields:
            Tuples of (parameter, score).
        """
        for param in param_lst:
            score = self._eval_score(param)
            yield param, score

    def _eval_parameters_parallel(self, param_lst: list[ParamType], *, n_workers: int) \
            -> Iterable[tuple[ParamType, float]]:
        """
        Evaluate all parameters in parallel using multiple processes.

        Args:
            param_lst: List of parameter objects to evaluate.
            n_workers: Number of worker processes.

        Yields:
            Tuples of (parameter, score).
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self._eval_score, p): p for p in param_lst}
            for future in concurrent.futures.as_completed(futures):
                param = futures[future]
                score = future.result()
                yield param, score

    def _run_search(self, *, n_workers: int) -> None:
        """
        Run the grid search over all parameter combinations.

        Args:
            n_workers: Number of parallel workers to use for evaluation.

        Raises:
            RuntimeError: If no valid parameter is found.
        """
        param_lst = self._list_param_objects()

        # Choose evaluation method based on the number of workers
        if n_workers > 1:
            it = self._eval_parameters_parallel(param_lst, n_workers=n_workers)
        else:
            it = self._eval_parameters_single(param_lst)
        bar = tqdm(
            it,
            total=len(param_lst),
        )
        for param, score in bar:
            self.add_record(param, score)
            bar.set_description(f"Best score={self.best_score:.3f}")

        if self.is_empty:
            raise RuntimeError("No valid parameter found")
