from ophyd import Component as Cpt, Device, Signal as UnusableSignal
from bluesky.utils import ancestry, share_ancestor, separate_devices


class Signal(UnusableSignal):
    def __init__(self, _, *, name=None, parent=None):
        super().__init__(name=name, parent=parent)



class A(Device):
    s1 = Cpt(Signal, '')
    s2 = Cpt(Signal, '')


class B(Device):
    a1 = Cpt(A, '')
    a2 = Cpt(A, '')


def test_ancestry():
    b = B('')
    assert ancestry(b) == [b]
    assert ancestry(b.a1) == [b.a1, b]
    assert ancestry(b.a1.s1) == [b.a1.s1, b.a1, b]

def test_share_ancestor():
    b1 = B('')
    b2 = B('')
    assert share_ancestor(b1, b1)
    assert share_ancestor(b1, b1.a1)
    assert share_ancestor(b1, b1.a1.s1)
    assert not share_ancestor(b2, b1)
    assert not share_ancestor(b2, b1.a1)
    assert not share_ancestor(b2, b1.a1.s1)


def test_separate_devices():
    b1 = B('')
    b2 = B('')
    a = A('')
    assert separate_devices([b1, b1.a1]) == [b1]
    assert separate_devices([b1.a1, b1.a2]) == [b1.a1, b1.a2]
    assert separate_devices([b1, b2.a1]) == [b1, b2.a1]
    assert separate_devices([b1.a1, b2.a2]) == [b1.a1, b2.a2]
    assert separate_devices([b1, a]) == [b1, a]
