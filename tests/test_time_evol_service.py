from functools import cache
from typing import Callable, Sequence

import numpy as np
from qutip.typing import ElementType

from core.fullstate import fullstate
from core.seed_or_rng import check_seed_or_rng_and_get_rng
from core.time_evol_solver import AbstractTimeEvolutionSolver, create_time_evol_solver
from model.axis import Axis
from model.physical_system import EachSingleQubitSingleAxisObservable, FullInteraction, \
    ResponsiveMagneticInteraction, AbstractResponsivePhysicalSystem
from service.compute_time_evol import ComputeTimeEvolStateSeriesDividedForwardService, \
    ComputeTimeEvolStateSeriesSingleForwardService
from service.dataset import DelayedSineDatasetGenerator

_N_QUBIT = 4
_OBSERVABLE_AXIS = Axis.Z
_J_AXIS = Axis.X
_J_MEAN = 0.0
_J_STD = 1.0
_H_AXIS = Axis.Z
_H_MEAN = 0.0
_H_STD = 1.0
_SEED = 0
_N_MPX = 3


class TestPhysicalSystem(AbstractResponsivePhysicalSystem):
    def __init__(self, seed: int):
        self._rng = check_seed_or_rng_and_get_rng(seed=seed)

        self._magnetic_interaction = ResponsiveMagneticInteraction.create_instance(
            n_qubit=_N_QUBIT,
            mean=_H_MEAN,
            std=_H_STD,
            axis=_H_AXIS,
            rng=self._rng,
        )
        self._full_interaction = FullInteraction.create_instance(
            n_qubit=_N_QUBIT,
            mean=_J_MEAN,
            std=_J_STD,
            axis=_J_AXIS,
            rng=self._rng,
        )

    @cache
    def create_hamiltonian(self, *, u_t: Callable[[float], float]) -> Sequence[ElementType]:
        return self._magnetic_interaction.create_hamiltonian() \
            + self._full_interaction.create_hamiltonian()


def _create_solver() -> AbstractTimeEvolutionSolver:
    return create_time_evol_solver(
        system=TestPhysicalSystem(seed=_SEED),
        observable=EachSingleQubitSingleAxisObservable(
            n_qubit=_N_QUBIT,
            axis=_OBSERVABLE_AXIS,
        ),
        init_psi=fullstate(",".join(["z+"] * _N_QUBIT)),
        backend="qutip-mesolve",
    )


_DS = DelayedSineDatasetGenerator(
    t_max=1.0,
    t_step=0.01,
    freq=1.0,
    phase_offset=0.0,
    discrete_lag=0.0,
    amplitude=1.0,
).create()


def test_time_evol_state_series_forward_service():
    service_div = ComputeTimeEvolStateSeriesDividedForwardService()
    valid_time_mask_div, state_series_div = service_div.execute(
        solver=_create_solver(),
        n_mpx=_N_MPX,
        u_t=_DS.u_t,
        t_arr=_DS.t_seq,
        reset_state=True,
    )
    assert state_series_div.n_state == _N_MPX * _N_QUBIT

    service_single = ComputeTimeEvolStateSeriesSingleForwardService()
    valid_time_mask_single, state_series_single = service_single.execute(
        solver=_create_solver(),
        n_mpx=_N_MPX,
        u_t=_DS.u_t,
        t_arr=_DS.t_seq,
        reset_state=True,
    )
    assert state_series_single.n_state == _N_MPX * _N_QUBIT

    mask = valid_time_mask_div & valid_time_mask_single
    index = np.where(mask)[0]
    for i in range(_N_MPX * _N_QUBIT):
        np.testing.assert_allclose(
            state_series_div[index, i],
            state_series_single[index, i],
            atol=1e-4,
        )
