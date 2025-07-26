from enum import IntEnum


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2

    @property
    def index(self) -> int:
        return self.value

    @property
    def name(self) -> str:
        return "XYZ"[self.index]
