from abc import ABC, abstractmethod
from typing import Sequence

from experiment.estimator.dto import StateComputationResult
from model.dataset import Dataset
from model.prediction_result import PredictionResult


class AbstractEstimator(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def fit(
            self,
            datasets: Sequence[Dataset],
            *,
            show_progress=False,
    ) -> list[PredictionResult]:
        raise NotImplementedError()

    @abstractmethod
    def predict(
            self,
            datasets: Sequence[Dataset],
            *,
            show_progress=False,
            state_comp_result: Sequence[StateComputationResult | None] | None = None,
    ) -> list[PredictionResult]:
        raise NotImplementedError()
