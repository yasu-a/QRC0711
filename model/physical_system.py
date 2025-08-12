import inspect
from abc import ABC, abstractmethod
from functools import cache, reduce
from typing import Callable, Sequence

import numpy as np
import qutip
from qutip.typing import ElementType

from core.fullgate import fullgate
from core.seed_or_rng import check_seed_or_rng_and_get_rng
from model.axis import Axis


class AbstractPhysicalObject(ABC):
    """
    物理システムの抽象基底クラス。

    量子リザバーコンピューティングで使用される物理システムの
    基本的なインターフェースを定義する。
    """

    _kind = ...  # kind of the physical object

    def __add__(self, other):
        """
        2つの物理システムを結合する。

        Args:
            other: 結合する物理システム

        Returns:
            ChainedPhysicalSystem: 結合された物理システム
        """
        if isinstance(other, AbstractPhysicalObject):
            return ChainedPhysicalSystem([self, other])
        return NotImplemented()


class AbstractIsolatedPhysicalSystem(AbstractPhysicalObject, ABC):
    """
    孤立した物理システムの抽象基底クラス。
    """

    _kind = "physical-system"

    @abstractmethod
    def create_hamiltonian(self) -> Sequence[ElementType]:
        """
        システムのハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: ハミルトニアンの項のリスト
        """
        raise NotImplementedError()


class AbstractResponsivePhysicalSystem(AbstractPhysicalObject, ABC):
    """
    ハミルトニアン生成時に入力を受け取る物理システムの抽象基底クラス。
    """

    _kind = "physical-system"

    @abstractmethod
    def create_hamiltonian(self, *, u_t: Callable[[float], float]) -> Sequence[ElementType]:
        """
        システムのハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: ハミルトニアンの項のリスト
        """
        raise NotImplementedError()


class AbstractObservable(AbstractPhysicalObject, ABC):
    """
    観測可能量の抽象基底クラス。量子システムで観測可能な物理量を表現する。
    """
    _kind = "observable"

    @abstractmethod
    def create_hamiltonian(self) -> Sequence[ElementType]:
        """
        システムのハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: ハミルトニアンの項のリスト
        """
        raise NotImplementedError()


class AbstractCollapseOperator(AbstractPhysicalObject, ABC):
    """
    崩壊演算子の抽象基底クラス。量子システムの非エルミート的な時間発展や散逸過程を表現する崩壊演算子を定義する。
    """
    _kind = "collapse-operator"

    @abstractmethod
    def create_hamiltonian(self) -> Sequence[ElementType]:
        """
        システムのハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: ハミルトニアンの項のリスト
        """
        raise NotImplementedError()


class ChainedPhysicalSystem(AbstractPhysicalObject):
    """
    複数の物理システムを結合したシステム。

    複数の物理システムのハミルトニアンを統合して、
    一つの複合システムとして扱う。
    """

    def __init__(self, chain: list[AbstractPhysicalObject], *, _check_kind_only_last=False):
        """
        結合物理システムを初期化する。

        Args:
            chain (list[AbstractPhysicalObject]): 結合する物理システムのリスト
        """
        self._chain = chain.copy()
        if len(chain) >= 2:
            # For efficiency, only compare the first and last elements if _check_kind_only_last is True
            if _check_kind_only_last:
                fail = chain[0]._kind != chain[-1]._kind
                if not isinstance(chain[-1], AbstractPhysicalObject):
                    raise TypeError(
                        f"The elements of the chain must be AbstractPhysicalObject, but got {type(chain[-1])}")
            else:
                fail = any(obj._kind != chain[0]._kind for obj in chain[1:])
                for obj in chain:
                    if not isinstance(obj, AbstractPhysicalObject):
                        raise TypeError(
                            f"The elements of the chain must be AbstractPhysicalObject, but got {type(obj)}")
            if fail:
                raise ValueError(
                    f"Chained physical system must have the same kind, but get {set(c._kind for c in chain)}"
                )
        self._kind = chain[0]._kind

    @cache
    def _get_kwarg_names(self, index: int) -> frozenset[str]:
        target = self._chain[index]
        # FIXME: protocolを上手く使ってtargetがcreate_hamiltonianを持っていることを記述し、type: ignoreを外す
        sig = inspect.signature(target.create_hamiltonian)  # type: ignore
        kwargs_names = [
            name for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        return frozenset(kwargs_names)

    @cache
    def _get_all_kwarg_names(self) -> frozenset[str]:
        kwargs_names = set()
        for i in range(len(self._chain)):
            kwargs_names |= self._get_kwarg_names(i)
        return frozenset(kwargs_names)

    def create_hamiltonian(self, **kwargs) -> Sequence[ElementType]:
        """
        結合されたシステムのハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj]: 全ての結合システムのハミルトニアン項のリスト
        """
        if frozenset(kwargs.keys()) != self._get_all_kwarg_names():
            raise ValueError(
                "Provided kwargs are not compatible with the chained system. "
                f"Expected: {sorted(self._get_all_kwarg_names())}, "
                f"Provided: {sorted(kwargs.keys())}"
            )

        ham = []
        for i, s in enumerate(self._chain):
            current_kwargs = {
                k: v for k, v in kwargs.items() if k in self._get_kwarg_names(i)}
            # FIXME: protocolを上手く使ってtargetがcreate_hamiltonianを持っていることを記述し、type: ignoreを外す
            ham.extend(s.create_hamiltonian(**current_kwargs))  # type: ignore
        return ham

    def __add__(self, other):
        if isinstance(other, ChainedPhysicalSystem):
            return type(self)([*self._chain, other._chain], _check_kind_only_last=True)
        elif isinstance(other, AbstractPhysicalObject):
            return type(self)([*self._chain, other], _check_kind_only_last=True)
        else:
            return NotImplemented


class EachSingleQubitSingleAxisObservable(AbstractObservable):
    """
    単一量子ビット・単一軸の観測可能量。

    指定した軸方向のパウリ演算子による観測量を
    各量子ビットに対して定義する。
    """

    def __init__(self, *, n_qubit: int, axis: Axis):
        """
        単一軸観測量を初期化する。

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


class TotalMagnetizationObservable(AbstractObservable):
    def __init__(self, *, n_qubit: int, axis: Axis):
        """
        単一軸観測量を初期化する。

        Args:
            n_qubit (int): 量子ビット数
            axis (Axis): 観測軸（X、Y、またはZ軸）
        """
        self._n_qubit = n_qubit
        self._axis = axis

    def create_hamiltonian(self):
        ham = reduce(
            lambda x, y: x + y,
            (
                fullgate(self._n_qubit, f"{self._axis.name}{i}")
                for i in range(self._n_qubit)
            ),
        )
        return [ham]


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


class FullInteraction(AbstractIsolatedPhysicalSystem):
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
    def _n_qubit(self):
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
            self._coeff[i][j] * fullgate(self._n_qubit,
                                         f"{self._axis.name}{i},{self._axis.name}{j}")
            for i in range(self._n_qubit - 1)
            for j in range(i + 1, self._n_qubit)
        ]


class MagneticInteractionMixin:
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
            ResponsiveMagneticInteraction: 生成されたインスタンス
        """
        assert n_qubit >= 1
        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)
        values = rng.noncentral_chisquare(
            df=1, nonc=(mean / std) ** 2, size=n_qubit)
        values *= rng.choice([-1, +1], size=len(values))
        coeff = np.array(values, dtype=np.float32)
        return cls(coeff, axis=axis)

    @property
    def _n_qubit(self):
        """量子ビット数を返す。"""
        return len(self._coeff)


class ResponsiveMagneticInteraction(MagneticInteractionMixin, AbstractResponsivePhysicalSystem):
    """
    磁場による相互作用。

    各量子ビットに対する磁場の効果を表現し、
    hi * σ_i * u(t) の形の項を生成する。
    時間依存性を持つことができる。
    """

    # hi * u(t) * fullgate([[axis, i]] foreach i)

    def create_hamiltonian(self, *, u_t: Callable[[float], float]) -> Sequence[ElementType]:
        """
        磁場相互作用のハミルトニアンを生成する。

        Args:
            u_t (Callable[[float], float], optional):
                時間依存性を表す関数。Noneの場合は時間非依存。

        Returns:
            list[qutip.Qobj] | list[list]:
                時間非依存の場合は演算子のリスト、
                時間依存の場合は [演算子, 時間関数] のペアのリスト
        """
        # Create a list of Hamiltonian for each qubit, scaled by coefficients
        ham_lst = [
            self._coeff[i] * fullgate(self._n_qubit, f"{self._axis.name}{i}")
            for i in range(self._n_qubit)
        ]

        # If time-dependent coefficient provided, wrap Hamiltonian in tuples with u_t
        ham_lst = [
            [ham, u_t]
            for ham in ham_lst
        ]
        return ham_lst


class IsolatedMagneticInteraction(MagneticInteractionMixin, AbstractIsolatedPhysicalSystem):
    """
    孤立した磁場相互作用。

    各量子ビットに対する磁場の効果を表現し、
    hi * σ_i の形の項を生成する。
    """

    def create_hamiltonian(self) -> Sequence[ElementType]:
        """
        磁場相互作用のハミルトニアンを生成する。

        Returns:
            list[qutip.Qobj] | list[list]:
                時間非依存の場合は演算子のリスト、
                時間依存の場合は [演算子, 時間関数] のペアのリスト
        """
        # Create a list of Hamiltonian for each qubit, scaled by coefficients
        ham_lst = [
            self._coeff[i] * fullgate(self._n_qubit, f"{self._axis.name}{i}")
            for i in range(self._n_qubit)
        ]
        return ham_lst


class NVPhysicalSystem(AbstractResponsivePhysicalSystem):
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
            j_axis: Axis | Sequence[Axis],
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

        # Full interaction
        if isinstance(j_axis, Axis):
            j_axis = [j_axis]
        assert all(isinstance(item, Axis) for item in j_axis)
        self._full_interaction = reduce(
            lambda x, y: x + y,
            (
                FullInteraction.create_instance(
                    n_qubit=n_qubit,
                    mean=j_mean / (n_qubit * (n_qubit - 1) / 2) / len(j_axis),  # (nC2*軸数)個の相互作用
                    std=j_std,
                    axis=j_axis_item,
                    rng=rng,
                )
                for j_axis_item in j_axis
            ),
        )

        # Magnetic interaction
        self._magnetic_interaction = IsolatedMagneticInteraction.create_instance(
            n_qubit=n_qubit,
            # time_dependent_magnetic_interactionと合わせて2n個の作用
            mean=h_mean / (n_qubit * 2),
            std=h_std,
            axis=h_axis,
            rng=rng,
        )

        # Time-dependent magnetic interaction
        self._time_dependent_magnetic_interaction = ResponsiveMagneticInteraction.create_instance(
            n_qubit=n_qubit,
            mean=h_mean / (n_qubit * 2),  # magnetic_interactionと合わせて2n個の作用
            std=h_std,
            axis=h_td_axis,
            rng=rng,
        )

    def create_hamiltonian(self, *, u_t: Callable[[float], float]) -> Sequence[ElementType]:
        """
        NVリザバーシステムの完全なハミルトニアンを生成する。

        Args:
            u_t: 時間依存項の係数関数

        Returns:
            list[qutip.Qobj]: システム全体のハミルトニアン項のリスト
        """
        ham_1 = self._full_interaction.create_hamiltonian()
        ham_2 = self._magnetic_interaction.create_hamiltonian()
        ham_3 = self._time_dependent_magnetic_interaction.create_hamiltonian(u_t=u_t)
        return ham_1 + ham_2 + ham_3


class SingleNVSystem(AbstractResponsivePhysicalSystem):
    def __init__(self, *, h: float, axis: Axis):
        self._h = h
        self._axis = axis

        # Time-dependent magnetic interaction
        self._time_dependent_magnetic_interaction = ResponsiveMagneticInteraction(
            coeff=np.array([h]),
            axis=self._axis,
        )

    def create_hamiltonian(self, *, u_t: Callable[[float], float]) -> Sequence[ElementType]:
        return self._time_dependent_magnetic_interaction.create_hamiltonian(u_t=u_t)


class FNQRCPhysicsSystem(AbstractResponsivePhysicalSystem):
    """
    FN-QRC (Fully Networked Quantum Reservoir Computing) 物理システム。

    論文 "Harnessing Disordered-Ensemble Quantum Dynamics for Machine Learning"
    で提案されたモデルに基づく。

    特徴：
    - 全結合ランダムネットワーク
    - バーチャルノードによる時分割処理
    - パラメータ化された回転ゲート
    """

    def __init__(
            self,
            *,
            n_qubit: int,
            j_mean: float,
            j_std: float,
            theta_mean: float,
            theta_std: float,
            rotation_axis: str,
            seed: int | None = None,
            rng: np.random.RandomState | None = None,
    ):
        """
        FN-QRC物理システムを初期化する。

        Args:
            n_qubit (int): 量子ビット数
            j_mean (float): 結合強度の平均
            j_std (float): 結合強度の標準偏差
            theta_mean (float): 磁場強度の平均
            theta_std (float): 磁場強度の標準偏差
            rotation_axis (str): 回転軸（論文式16ではZ軸固定、互換性のため保持）
            seed (int | None, optional): 乱数シード
            rng (np.random.RandomState | None, optional): 乱数生成器
        """
        self._n_qubit = n_qubit
        self._rotation_axis = rotation_axis

        rng = check_seed_or_rng_and_get_rng(seed=seed, rng=rng)

        # 量子ビット間の全結合相互作用係数を生成
        # 上三角行列として結合係数を生成
        n_connections = n_qubit * (n_qubit - 1) // 2
        j_values = rng.normal(j_mean, j_std, size=n_connections)

        self._j_matrix = np.zeros((n_qubit, n_qubit))
        triu_indices = np.triu_indices(n_qubit, k=1)
        self._j_matrix[triu_indices] = j_values

        # 各量子ビットの磁場強度を生成（virtual nodeには依存しない）
        self._theta_values = rng.normal(theta_mean, theta_std, size=n_qubit)

    def create_hamiltonian(self, *, u_t: Callable[[float], float]) -> Sequence[ElementType]:
        """
        FN-QRCシステムのハミルトニアンを生成する。

        論文式16: H = Σ Ji Xi Xj + hZi に基づく
        物理系は固定され、virtual nodeには依存しない。

        Args:
            u_t (Callable[[float], float], optional): 時間依存の入力関数

        Returns:
            list[qutip.Qobj]: システムのハミルトニアン項のリスト
        """
        hamiltonian_terms = []

        # 量子ビット間の結合項（J_{ij} σ_i^x σ_j^x）論文式16
        for i in range(self._n_qubit):
            for j in range(i + 1, self._n_qubit):
                j_ij = self._j_matrix[i, j]
                if j_ij != 0:
                    h_term = j_ij * fullgate(self._n_qubit, f"X{i},X{j}")
                    hamiltonian_terms.append(h_term)

        # 時間依存の磁場項として [演算子, 時間関数] のペアを作成
        for k in range(self._n_qubit):
            h_k = self._theta_values[k]
            if h_k != 0:
                h_operator = h_k * fullgate(self._n_qubit, f"Z{k}")
                hamiltonian_terms.append([h_operator, u_t])

        return hamiltonian_terms

    @property
    def n_qubit(self) -> int:
        """量子ビット数を返す。"""
        return self._n_qubit

    @property
    def j_matrix(self) -> np.ndarray:
        """結合係数行列を返す。"""
        return self._j_matrix.copy()

    @property
    def theta_values(self) -> np.ndarray:
        """磁場強度配列を返す。"""
        return self._theta_values.copy()
