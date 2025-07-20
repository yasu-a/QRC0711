from functools import cache

import qutip

_MAT_MAPPING = {
    "x": qutip.sigmax(),
    "y": qutip.sigmay(),
    "z": qutip.sigmaz(),
    "i": qutip.qeye(2),
}


@cache
def fullgate(n_qubit: int, fmt: str):
    """Create a full quantum gate operator by tensor product of Pauli matrices.

    Args:
        n_qubit: Number of qubits in the system
        fmt: String format specifying gates in the form "G1,G2,...", where each G is:
             - First character: Gate type (X/Y/Z for Pauli matrices, I for identity)  
             - Remaining characters: Qubit index the gate acts on
             Gates must be ordered by increasing qubit index

    Returns:
        qutip.Qobj: Quantum operator representing the tensor product of specified gates,
                   with identity matrices filling unspecified positions

    Examples:
        >>> fullgate(2, "X1") # Identity on qubit 0, X gate on qubit 1
        >>> fullgate(3, "X0,Y1,Z2") # X, Y, Z gates on qubits 0, 1, 2 respectively

    Raises:
        ValueError: If qubit indices in fmt are not unique and ordered
    """

    config = []
    for item in fmt.split(","):
        item = item.strip()
        mat = _MAT_MAPPING[item[0].lower()]
        qubit_index = int(item[1:])
        config.append((qubit_index, mat))

    import itertools
    if not all(c1[0] < c2[0] for c1, c2 in itertools.pairwise(config)):
        raise ValueError(f"all indexes must be unique and ordered: {fmt=}")
    if not all(0 <= c[0] < n_qubit for c in config):
        raise ValueError(
            f"index is out of valid range [0, n_qubit - 1] in format: {fmt}")

    import collections
    qubit_index_to_mat: dict[int, qutip.Qobj] = collections.defaultdict(lambda: _MAT_MAPPING["i"])
    for qubit_index, mat in config:
        qubit_index_to_mat[qubit_index] = mat

    return qutip.tensor(*(qubit_index_to_mat[i] for i in range(n_qubit)))
