import copy
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Sequence, Any

ParamType = TypeVar("ParamType")


class AbstractParameterSearcher(ABC, Generic[ParamType]):
    """
    Abstract base class for parameter searchers.

    This class provides a framework for searching over a parameter space,
    evaluating parameter sets using a scoring function, and keeping track of
    search history and constraints.
    """

    def __init__(
            self,
            scorer: Callable[[ParamType], float],
            param_grid: dict[str, Sequence[Any]],
            param_mapper: Callable[[dict[str, Any]], ParamType],
            constraint_predicate: Callable[[ParamType], bool] | None = None,
    ):
        """
        Initialize the parameter searcher.

        Args:
            scorer: A callable that takes a parameter set and returns a float score.
            param_grid: A dictionary mapping parameter names to sequences of possible values.
            param_mapper: A callable that maps a dictionary of parameter values to a ParamType instance.
            constraint_predicate: An optional callable that checks if a parameter set is feasible.
        """
        self.__scorer = scorer
        self._param_grid = param_grid  # TODO: make this private
        self._param_mapper = param_mapper  # TODO: make this private
        self.__constraint_predicate = constraint_predicate

        self.__history: list[tuple[ParamType, float]] = []  # List of (parameter, score) tuples

    @property
    def is_empty(self) -> bool:
        """
        Returns True if no search history is recorded.
        """
        return len(self.__history) == 0

    @property
    def best_score(self) -> float:
        """
        Returns the best score found so far.

        Raises:
            ValueError: If no search history is recorded.
        """
        if not self.__history:
            raise ValueError("no search history recorded")
        return max(score for _, score in self.__history)

    @property
    def best_param(self) -> ParamType:
        """
        Returns the parameter set with the best score.

        Raises:
            ValueError: If no search history is recorded.
        """
        if not self.__history:
            raise ValueError("no search history recorded")
        return max(self.__history, key=lambda x: x[1])[0]

    @property
    def history(self) -> list[tuple[ParamType, float]]:
        """
        Returns a deep copy of the search history as a list of (parameter, score) tuples.
        """
        return copy.deepcopy(self.__history)

    def add_record(self, p: ParamType, score: float) -> None:  # TODO: make this protected
        """
        Add a parameter set and its score to the search history.

        Args:
            p: The parameter set.
            score: The score associated with the parameter set.
        """
        self.__history.append((p, score))

    def _eval_score(self, p: ParamType) -> float:
        """
        Evaluate the score for a given parameter set.

        Args:
            p: The parameter set.

        Returns:
            The score as a float.

        Raises:
            AssertionError: If the scorer does not return a float.
        """
        score = self.__scorer(p)
        assert isinstance(score, float), (type(score), score)
        return score

    def _is_feasible(self, p: ParamType) -> bool:
        """
        Check if the parameter set is within the feasible region.

        Args:
            p: The parameter set.

        Returns:
            True if the parameter set is feasible, False otherwise.
        """
        if self.__constraint_predicate is None:
            return True
        return self.__constraint_predicate(p)

    @abstractmethod
    def _run_search(self, *, n_workers: int) -> None:
        """
        Run the search algorithm.

        Procedures to be implemented in subclasses (see e.g. GridParameterSearcher, GAParameterSearcher):
        - Generate candidate parameter sets from the search space (e.g., grid enumeration, random sampling, or population generation for GA).
        - Evaluate the score for each candidate parameter (sequentially or in parallel).
        - Record each parameter and its score using `add_record`.
        - For evolutionary algorithms, perform population evolution (e.g., selection, crossover, mutation) as needed.
        - Track and update the best parameter and score using `best_param` and `best_score`.
        - Raise an exception if no valid parameter is found.

        Args:
            n_workers: The number of worker processes or threads to use for evaluation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def search(self, *, n_workers: int) -> tuple[ParamType, float]:
        """
        Run the search and return the best parameter set and its score.

        Args:
            n_workers: The number of worker processes or threads to use.

        Returns:
            A tuple containing the best parameter set and its score.
        """
        self._run_search(n_workers=n_workers)
        return self.best_param, self.best_score
