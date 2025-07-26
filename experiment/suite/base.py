from abc import ABC, abstractmethod


class AbstractExperimentSuite(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError()
