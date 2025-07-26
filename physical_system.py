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
    """
    物理システムの抽象基底クラス。
    
    量子リザバーコンピューティングで使用される物理システムの
    基本的なインターフェースを定義する。
    """
    _kind = "physical-system"

    @abstractmethod
    def create_hamiltonian(self, **kwargs) -> list[qutip.Qobj]:
        """
        システムのハミルトニアンを生成する。

        Args:
            **kwargs: ハミルトニアン生成に必要な追加パラメータ

        Returns:
            list[qutip.Qobj]: ハミルトニアンの項のリスト
        """
        raise NotImplementedError()

    def __add__(self, other):
        """
        2つの物理システムを結合する。

        Args:
            other: 結合する他の物理システム

        Returns:
            結合されたChainedPhysicalSystemまたは元のシステムのコピー
        """
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
    """
    複数の物理システムを結合したシステム。
    
    複数の物理システムのハミルトニアンを統合して、
    一つの複合システムとして扱う。
    """
    _kind = None

    def __init__(self, chain: list[AbstractPhysicalSystem]):
        """
        結合物理システムを初期化する。

        Args:
            chain (list[AbstractPhysicalSystem]): 結合する物理システムのリスト
        """
        self._chain = chain

    def create_hamiltonian(self) -> list[qutip.Qobj]:
        """
        結合されたシステムのハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: 全ての結合システムのハミルトニアン項のリスト
        """
        # TODO: chained system cannot create hamiltonian with kwargs
        ham = []
        for s in self._chain:
            ham.extend(s.create_hamiltonian())
        return ham

    def __add__(self, other):
        """
        他のシステムとさらに結合する。

        Args:
            other: 追加で結合するシステム

        Returns:
            拡張されたChainedPhysicalSystem
        """
        if other is None:
            return ChainedPhysicalSystem(self._chain)
        elif isinstance(other, ChainedPhysicalSystem):
            return ChainedPhysicalSystem(self._chain + other._chain)
        elif isinstance(other, AbstractPhysicalSystem):
            return ChainedPhysicalSystem(self._chain + [other])
        else:
            return NotImplemented


class AbstractObservable(AbstractPhysicalSystem, ABC):
    """
    観測可能量の抽象基底クラス。
    
    量子システムで観測可能な物理量を表現する。
    """
    _kind = "observable"


class AbstractCollapseOperator(AbstractPhysicalSystem, ABC):
    """
    崩壊演算子の抽象基底クラス。
    
    量子システムの非エルミート的な時間発展や
    散逸過程を表現する崩壊演算子を定義する。
    """
    _kind = "collapse-operator"


class NVReservoirObservable(AbstractObservable):
    """
    NVセンター量子リザバーの観測可能量。
    
    指定した軸方向のパウリ演算子による観測量を
    各量子ビットに対して定義する。
    """

    def __init__(self, *, n_qubit: int, axis: Axis):
        """
        NVリザバー観測量を初期化する。

        Args:
            n_qubit (int): 量子ビット数
            axis (Axis): 観測軸（X、Y、またはZ軸）
        """
        self._n_qubit = n_qubit
        self._axis = axis

    def create_hamiltonian(self):
        """
        観測可能量のハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: 各量子ビットの指定軸パウリ演算子のリスト
        """
        return [
            fullgate(self._n_qubit, f"{self._axis.name}{i}")
            for i in range(self._n_qubit)
        ]


class NVReservoirCollapseOperator(AbstractCollapseOperator):
    """
    NVセンター量子リザバーの崩壊演算子。
    
    Z軸方向の散逸過程を表現する崩壊演算子を
    各量子ビットに対して定義する。
    """

    def __init__(self, *, n_qubit: int, gamma_z: float):
        """
        NVリザバー崩壊演算子を初期化する。

        Args:
            n_qubit (int): 量子ビット数
            gamma_z (float): Z軸方向の散逸レート
        """
        self._n_qubit = n_qubit
        self._gamma_z = gamma_z

    def create_hamiltonian(self):
        """
        崩壊演算子のハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: 各量子ビットのZ軸崩壊演算子のリスト
        """
        return [
            self._gamma_z ** .5 * fullgate(self._n_qubit, f"Z{i}")
            for i in range(self._n_qubit)
        ]


class FullInteraction(AbstractPhysicalSystem):
    """
    量子ビット間の全結合相互作用。
    
    全ての量子ビットペア間の相互作用を表現し、
    Jij * σ_i ⊗ σ_j の形の項を生成する。
    """
    # Jij * fullgate([[axis, i], [axis, j]] foreach i, j where i < j)

    def __init__(self, coeff: np.ndarray, *, axis: Axis):
        """
        全結合相互作用を初期化する。

        Args:
            coeff (np.ndarray): 相互作用係数の行列（上三角のみ有効）
            axis (Axis): 相互作用の軸方向
        """
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
        """
        ランダムな相互作用係数を持つインスタンスを生成する。

        Args:
            n_qubit (int): 量子ビット数
            mean (float): 相互作用強度の平均値
            std (float): 相互作用強度の標準偏差
            axis (Axis): 相互作用の軸方向
            seed (int | None, optional): 乱数シード
            rng (np.random.RandomState | None, optional): 乱数生成器

        Returns:
            FullInteraction: 生成されたインスタンス
        """
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
        """量子ビット数を返す。"""
        return len(self._coeff)

    @cache
    def create_hamiltonian(self):
        """
        全結合相互作用のハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: 各量子ビットペアの相互作用項のリスト
        """
        return [
            self._coeff[i][j] * fullgate(self.n_qubit, f"{self._axis.name}{i},{self._axis.name}{j}")
            for i in range(self.n_qubit - 1)
            for j in range(i + 1, self.n_qubit)
        ]


class MagneticInteraction(AbstractPhysicalSystem):
    """
    磁場による相互作用。
    
    各量子ビットに対する磁場の効果を表現し、
    hi * σ_i * u(t) の形の項を生成する。
    時間依存性を持つことができる。
    """
    # hi * fullgate([[axis, i]] foreach i) * u_t(t)

    def __init__(self, coeff: np.ndarray, *, axis: Axis):
        """
        磁場相互作用を初期化する。

        Args:
            coeff (np.ndarray): 各量子ビットの磁場強度係数
            axis (Axis): 磁場の軸方向
        """
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
        """
        ランダムな磁場強度を持つインスタンスを生成する。

        Args:
            n_qubit (int): 量子ビット数
            mean (float): 磁場強度の平均値
            std (float): 磁場強度の標準偏差
            axis (Axis): 磁場の軸方向
            seed (int | None, optional): 乱数シード
            rng (np.random.RandomState | None, optional): 乱数生成器

        Returns:
            MagneticInteraction: 生成されたインスタンス
        """
        assert n_qubit >= 1
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
        values = rng.noncentral_chisquare(df=1, nonc=(mean / std) ** 2, size=n_qubit)
        values *= rng.choice([-1, +1], size=len(values))
        coeff = np.array(values, dtype=np.float32)
        return cls(coeff, axis=axis)

    @property
    def n_qubit(self):
        """量子ビット数を返す。"""
        return len(self._coeff)

    def create_hamiltonian(self, *, td_coeff: Callable[[float], float] = None):
        """
        磁場相互作用のハミルトニアンを生成する。

        Args:
            td_coeff (Callable[[float], float], optional): 
                時間依存性を表す関数。Noneの場合は時間非依存。

        Returns:
            list[qutip.Qobj] | list[list]: 
                時間非依存の場合は演算子のリスト、
                時間依存の場合は [演算子, 時間関数] のペアのリスト
        """
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
    """
    NVセンターベースの量子リザバー物理システム。
    
    以下の相互作用を組み合わせた完全なQRCシステム：
    - 量子ビット間の全結合相互作用
    - 静的磁場による相互作用
    - 時間依存磁場による駆動
    """

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
        """
        NVリザバー物理システムを初期化する。

        Args:
            n_qubit (int): 量子ビット数
            j_mean (float): 量子ビット間相互作用強度の平均
            j_std (float): 量子ビット間相互作用強度の標準偏差
            j_axis (Axis): 量子ビット間相互作用の軸
            h_mean (float): 磁場強度の平均
            h_std (float): 磁場強度の標準偏差
            h_axis (Axis): 静的磁場の軸
            h_td_axis (Axis): 時間依存磁場の軸
            seed (int | None, optional): 乱数シード
            rng (np.random.RandomState | None, optional): 乱数生成器
        """
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
        """
        NVリザバーシステムの完全なハミルトニアンを生成する。

        Args:
            td_coeff: 時間依存項の係数関数

        Returns:
            list[qutip.Qobj]: システム全体のハミルトニアン項のリスト
        """
        ham_1 = self._full_interaction.create_hamiltonian()
        ham_2 = self._magnetic_interaction.create_hamiltonian()
        ham_3 = self._time_dependent_magnetic_interaction.create_hamiltonian(td_coeff=td_coeff)
        return ham_1 + ham_2 + ham_3
