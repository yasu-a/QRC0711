import pytest

from utils.axis import Axis


@pytest.mark.parametrize(
    ("axis", "index"),
    [
        (Axis.X, 0),
        (Axis.Y, 1),
        (Axis.Z, 2),
    ]
)
def test_axis_index(axis, index):
    assert axis.index == index


@pytest.mark.parametrize(
    ("axis", "name"),
    [
        (Axis.X, "X"),
        (Axis.Y, "Y"),
        (Axis.Z, "Z"),
    ]
)
def test_axis_name(axis, name):
    assert axis.name == name
