from ophyd import Component as Cpt, Device, SoftSignal
from bluesky.utils import ancestry, separate_devices


class A(Device):
    s1 = Cpt(SoftSignal, '')
    s2 = Cpt(SoftSignal, '')


class B(Device):
    a = Cpt(A, '')


def test_ancestry():
    b = B()
    assert ancestry(b) == [b]
    assert ancestry(b.a) == [b.a, b]
    assert ancestry(b.a.s1) == [b.a.s1, b.a, b]
