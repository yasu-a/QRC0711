import copy
from abc import ABC, abstractmethod
from functools import cache
from typing import Callable

import numpy as np
import qutip

from utils import fullgate
from utils.axis import Axis
from utils.seed_or_rng import check_seed_or_rng_and_get_rng


class AbstractPhysicalSystem(ABC):
    _kind = "physical-system"

    @abstractmethod
    def create_hamiltonian(self, **kwargs) -> list[qutip.Qobj]:
        raise NotImplementedError()

    def __add__(self, other):
        if other is None:
            return copy.deepcopy(self)
        if not isinstance(other, AbstractPhysicalSystem):
            return NotImplemented
        if self._kind is None or other._kind is None:
            return NotImplemented
        if self._kind != other._kind:
            return NotImplemented
        return ChainedPhysicalSystem([self, other])


class ChainedPhysicalSystem(AbstractPhysicalSystem):
    _kind = None

    def __init__(self, chain: list[AbstractPhysicalSystem]):
        self._chain = chain

    def create_hamiltonian(self) -> list[qutip.Qobj]:
        # TODO: chained system cannot create hamiltonian with kwargs
        ham = []
        for s in self._chain:
            ham.extend(s.create_hamiltonian())
        return ham

    def __add__(self, other):
        if other is None:
            return ChainedPhysicalSystem(self._chain)
        elif isinstance(other, ChainedPhysicalSystem):
            return ChainedPhysicalSystem(self._chain + other._chain)
        elif isinstance(other, AbstractPhysicalSystem):
            return ChainedPhysicalSystem(self._chain + [other])
        else:
            return NotImplemented


class AbstractObservable(AbstractPhysicalSystem, ABC):
    _kind = "observable"


class AbstractCollapseOperator(AbstractPhysicalSystem, ABC):
    _kind = "collapse-operator"


class NVReservoirObservable(AbstractObservable):
    def __init__(self, *, n_qubit: int, axis: Axis):
        self._n_qubit = n_qubit
        self._axis = axis

    def create_hamiltonian(self):
        return [
            fullgate(self._n_qubit, f"{self._axis.name}{i}")
            for i in range(self._n_qubit)
        ]


class NVReservoirCollapseOperator(AbstractCollapseOperator):
    def __init__(self, *, n_qubit: int, gamma_z: float):
        self._n_qubit = n_qubit
        self._gamma_z = gamma_z

    def create_hamiltonian(self):
        return [
            self._gamma_z ** .5 * fullgate(self._n_qubit, f"Z{i}")
            for i in range(self._n_qubit)
        ]


class FullInteraction(AbstractPhysicalSystem):
    # Jij * fullgate([[axis, i], [axis, j]] foreach i, j where i < j)

    def __init__(self, coeff: np.ndarray, *, axis: Axis):
        assert coeff.ndim == 2 and coeff.shape[0] == coeff.shape[1], coeff.shape
        self._coeff = coeff
        self._axis = axis

    @classmethod
    def create_instance(
            cls,
            *,
            n_qubit: int, mean: float, std: float, axis: Axis,
            seed: int | None = None, rng: np.random.RandomState | None = None,
    ):
        assert n_qubit >= 1
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
        values = rng.noncentral_chisquare(df=1, nonc=(mean / std) ** 2,
                                          size=n_qubit * (n_qubit - 1) // 2)
        values *= rng.choice([-1, +1], size=len(values))
        coeff = np.full((n_qubit, n_qubit), np.nan, np.float32)
        row_indices, col_indices = np.triu_indices(n_qubit, k=1)
        coeff[row_indices, col_indices] = values
        return cls(coeff, axis=axis)

    @property
    def n_qubit(self):
        return len(self._coeff)

    @cache
    def create_hamiltonian(self):
        return [
            self._coeff[i][j] * fullgate(self.n_qubit, f"{self._axis.name}{i},{self._axis.name}{j}")
            for i in range(self.n_qubit - 1)
            for j in range(i + 1, self.n_qubit)
        ]


class MagneticInteraction(AbstractPhysicalSystem):
    # hi * fullgate([[axis, i]] foreach i) * u_t(t)

    def __init__(self, coeff: np.ndarray, *, axis: Axis):
        assert coeff.ndim == 1
        self._coeff = coeff
        self._axis = axis

    @classmethod
    def create_instance(
            cls,
            *,
            n_qubit: int, mean: float, std: float, axis: Axis,
            seed: int | None = None, rng: np.random.RandomState | None = None,
    ):
        assert n_qubit >= 1
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
        values = rng.noncentral_chisquare(df=1, nonc=(mean / std) ** 2, size=n_qubit)
        values *= rng.choice([-1, +1], size=len(values))
        coeff = np.array(values, dtype=np.float32)
        return cls(coeff, axis=axis)

    @property
    def n_qubit(self):
        return len(self._coeff)

    def create_hamiltonian(self, *, td_coeff: Callable[[float], float] = None):
        # Create a list of Hamiltonian for each qubit, scaled by coefficients
        ham_lst = [
            self._coeff[i] * fullgate(self.n_qubit, f"{self._axis.name}{i}")
            for i in range(self.n_qubit)
        ]

        # If time-dependent coefficient provided, wrap Hamiltonian in tuples with td_coeff
        if td_coeff is not None:
            ham_lst = [
                [ham, td_coeff]
                for ham in ham_lst
            ]
        return ham_lst


class NVReservoirPhysicsSystem(AbstractPhysicalSystem):
    def __init__(
            self,
            *,
            n_qubit: int,
            j_mean: float,
            j_std: float,
            j_axis: Axis,
            h_mean: float,
            h_std: float,
            h_axis: Axis,
            h_td_axis: Axis,
            seed: int | None = None,
            rng: np.random.RandomState | None = None,
    ):
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
        self._full_interaction = FullInteraction.create_instance(
            n_qubit=n_qubit,
            mean=j_mean / (n_qubit * (n_qubit - 1) / 2),  # nC2個の相互作用
            std=j_std,
            axis=j_axis,
            rng=rng,
        )
        self._magnetic_interaction = MagneticInteraction.create_instance(
            n_qubit=n_qubit,
            mean=h_mean / (n_qubit * 2),  # time_dependent_magnetic_interactionと合わせて2n個の作用
            std=h_std,
            axis=h_axis,
            rng=rng,
        )
        self._time_dependent_magnetic_interaction = MagneticInteraction.create_instance(
            n_qubit=n_qubit,
            mean=h_mean / (n_qubit * 2),  # magnetic_interactionと合わせて2n個の作用
            std=h_std,
            axis=h_td_axis,
            rng=rng,
        )

    def create_hamiltonian(self, *, td_coeff):
        ham_1 = self._full_interaction.create_hamiltonian()
        ham_2 = self._magnetic_interaction.create_hamiltonian()
        ham_3 = self._time_dependent_magnetic_interaction.create_hamiltonian(td_coeff=td_coeff)
        return ham_1 + ham_2 + ham_3
