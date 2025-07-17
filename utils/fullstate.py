from functools import cache

import qutip

# 量子ビットの基底状態を定義
_0 = qutip.basis(2, 0)  # |0> or z+
_1 = qutip.basis(2, 1)  # |1> or z-

# _VEC_MAPPING を拡張
_VEC_MAPPING = {
    "z+": _0,
    "z-": _1,
    "x+": (_0 + _1).unit(),  # ( |0> + |1> ) / sqrt(2)
    "x-": (_0 - _1).unit(),  # ( |0> - |1> ) / sqrt(2)
    "y+": (_0 + 1j * _1).unit(),  # ( |0> + i|1> ) / sqrt(2)
    "y-": (_0 - 1j * _1).unit(),  # ( |0> - i|1> ) / sqrt(2)
}


@cache
def fullstate(fmt: str) -> qutip.Qobj:
    vec_lst = []
    for item in fmt.split(","):
        item = item.strip()
        vec = _VEC_MAPPING[item.lower()]
        vec_lst.append(vec)
    return qutip.tensor(*vec_lst)
