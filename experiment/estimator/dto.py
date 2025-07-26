from dataclasses import dataclass

import numpy as np

from model.state_series import AbstractStateSeries


@dataclass(slots=True)
class StateComputationResult:
    t_arr: np.ndarray  # (T,)
    u_mlt_arr: np.ndarray  # (T, n_in)
    states: AbstractStateSeries  # length: T
    y_mlt_arr: np.ndarray  # (T, n_out)

    # noinspection DuplicatedCode
    def __post_init__(self):
        assert isinstance(self.t_arr, np.ndarray), (type(self.t_arr), self.t_arr)
        assert isinstance(self.u_mlt_arr, np.ndarray), (type(self.u_mlt_arr), self.u_mlt_arr)
        assert isinstance(self.states, AbstractStateSeries), (type(self.states), self.states)
        assert isinstance(self.y_mlt_arr, np.ndarray), (type(self.y_mlt_arr), self.y_mlt_arr)

        n_t = len(self.t_arr)
        assert self.t_arr.ndim == 1, self.t_arr.shape
        assert self.u_mlt_arr.ndim == 2, self.u_mlt_arr.shape
        assert self.u_mlt_arr.shape[0] == n_t, (self.u_mlt_arr.shape[0], n_t)
        assert len(self.states) == n_t, (len(self.states), n_t)
        assert self.y_mlt_arr.ndim == 2, self.y_mlt_arr.shape
        assert self.y_mlt_arr.shape[0] == n_t, (self.y_mlt_arr.shape[0], n_t)

        # copy arrays and make readonly
        self.t_arr = self.t_arr.copy()
        self.t_arr.setflags(write=False)
        self.u_mlt_arr = self.u_mlt_arr.copy()
        self.u_mlt_arr.setflags(write=False)
        self.y_mlt_arr = self.y_mlt_arr.copy()
        self.y_mlt_arr.setflags(write=False)
