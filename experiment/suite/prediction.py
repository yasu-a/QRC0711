from typing import Callable

import numpy as np

from experiment.estimator.base import AbstractEstimator
from experiment.suite.base import AbstractExperimentSuite
from model.prediction_result import PredictionResultSet
from service.dataset import AbstractDatasetGenerator


class PredictionExperimentSuite(AbstractExperimentSuite):
    """任意の予測実験Modelをまとめて管理・評価する汎用Suiteクラス"""

    def __init__(
            self,
            *,
            generator_fn: Callable[[np.random.RandomState], AbstractDatasetGenerator],
            n_train_samples: int,
            n_test_samples: int,
            rng: np.random.RandomState,
            model_class: type[AbstractEstimator],
            model_kwargs: dict,
            show_progress: bool = False,
    ):
        self._generator_fn = generator_fn
        self._n_train_samples = n_train_samples
        self._n_test_samples = n_test_samples
        self._rng = rng
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        self._show_progress = show_progress

        self._run = False

        self._results_train: PredictionResultSet | None = None
        self._results_test: PredictionResultSet | None = None

    def run(self):
        if self._run:
            raise ValueError(
                "run() method has already been called. Cannot run experiment multiple times."
            )

        # Generate training datasets
        datasets_train = []
        for _ in range(self._n_train_samples):
            dataset = self._generator_fn(self._rng).create()
            datasets_train.append(dataset)

        # Generate test datasets
        datasets_test = []
        for _ in range(self._n_test_samples):
            dataset = self._generator_fn(self._rng).create()
            datasets_test.append(dataset)

        # Create estimator
        model = self._model_class(**self._model_kwargs)

        # Create and fit model on training data
        results_train = model.fit(datasets_train, show_progress=self._show_progress)
        self._results_train = PredictionResultSet(results_train)

        # Get predictions on test data
        results_test = model.predict(datasets_test, show_progress=self._show_progress)
        self._results_test = PredictionResultSet(results_test)

        self._run = True

    def _check_run(self) -> None:
        if not self._run:
            raise RuntimeError("You must call run() before accessing results_train.")

    @property
    def results_train(self) -> PredictionResultSet:
        self._check_run()
        return self._results_train

    @property
    def results_test(self) -> PredictionResultSet:
        self._check_run()
        return self._results_test
