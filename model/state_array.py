from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class AbstractState2DArray(ABC):  # immutable
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_state(self) -> int:
        raise NotImplementedError()

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), self.n_state

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError()

    @abstractmethod
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """numpy配列として返す"""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key) -> "AbstractState2DArray | np.ndarray":
        raise NotImplementedError()

    def series(self, i: int) -> np.ndarray:
        return self[:, i]


@dataclass
class QRCStateTimeStep:
    def __init__(
            self,
            states: np.ndarray,  # shape: (V, 状態数)
    ):
        assert states.ndim == 2, states.shape
        self._states = states

    def __len__(self) -> int:
        return self._states.shape[0]

    @property
    def n_states_per_times(self) -> int:
        return self._states.shape[1]

    @property
    def n_state(self) -> int:
        return len(self) * self.n_states_per_times

    @property
    def state_vector(self) -> np.ndarray:
        v = self._states.flatten()
        assert len(v) == self.n_state
        return v


class QRCStateArray(AbstractState2DArray):  # immutable
    def __init__(
            self,
            steps: list[QRCStateTimeStep],
    ):
        assert steps, "steps cannot be empty"
        assert all(step.n_state == steps[0].n_state for step in steps[1:]), [step.n_state for step
                                                                             in steps]

        # stepsを配列に変換してからstepsを破棄
        self._data = np.array([step.state_vector for step in steps])
        self._data.setflags(write=False)  # immutableにする
        self._n_states = steps[0].n_state

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def n_state(self) -> int:
        return self._n_states

    @property
    def dtype(self):
        return self._data.dtype

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """numpy配列として返す"""
        if dtype is None:
            dtype = self._data.dtype
        if copy is None:
            copy = True
        return np.array(self._data, dtype=dtype, copy=copy)

    def __getitem__(self, key):
        return self._data[key]
