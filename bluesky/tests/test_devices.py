from collections import defaultdict

from bluesky.utils import ancestry, share_ancestor, separate_devices
from bluesky.plans import trigger_and_read
from bluesky import Msg
import pytest

# some module level globals.
ophyd = None
reason = ''

try:
    import ophyd
    from ophyd import Component as Cpt, Device, Signal as UnusableSignal
except ImportError as ie:
    # pytestmark = pytest.mark.skip
    ophyd = None
    reason = str(ie)
else:
    # define the classes only if ophyd is available
    class Signal(UnusableSignal):
        def __init__(self, _, *, name=None, parent=None):
            super().__init__(name=name, parent=parent)
    class A(Device):
        s1 = Cpt(Signal, '')
        s2 = Cpt(Signal, '')
    class B(Device):
        a1 = Cpt(A, '')
        a2 = Cpt(A, '')

# define a skip condition based on if ophyd is available or not
requires_ophyd = pytest.mark.skipif(ophyd is None, reason=reason)


@requires_ophyd
def test_ancestry():
    b = B('')
    assert ancestry(b) == [b]
    assert ancestry(b.a1) == [b.a1, b]
    assert ancestry(b.a1.s1) == [b.a1.s1, b.a1, b]

@requires_ophyd
def test_share_ancestor():
    b1 = B('')
    b2 = B('')
    assert share_ancestor(b1, b1)
    assert share_ancestor(b1, b1.a1)
    assert share_ancestor(b1, b1.a1.s1)
    assert not share_ancestor(b2, b1)
    assert not share_ancestor(b2, b1.a1)
    assert not share_ancestor(b2, b1.a1.s1)


@requires_ophyd
def test_separate_devices():
    b1 = B('')
    b2 = B('')
    a = A('')
    assert separate_devices([b1, b1.a1]) == [b1]
    assert separate_devices([b1.a1, b1.a2]) == [b1.a1, b1.a2]
    assert separate_devices([b1, b2.a1]) == [b1, b2.a1]
    assert separate_devices([b1.a1, b2.a2]) == [b1.a1, b2.a2]
    assert separate_devices([b1, a]) == [b1, a]


@requires_ophyd
def test_monitor(fresh_RE):
    docs = []
    def collect(name, doc):
        docs.append(doc)

    a = A('')
    def plan():
        yield Msg('open_run')
        yield Msg('monitor', a.s1)
        a.s1._run_subs(sub_type='value')
        yield Msg('close_run')
    fresh_RE(plan(), collect)
    assert len(docs) == 4


@requires_ophyd
def test_monitor_with_pause_resume(fresh_RE):
    docs = []
    def collect(name, doc):
        docs.append(doc)

    a = A('')
    def plan():
        yield Msg('open_run')
        yield Msg('monitor', a.s1)
        yield Msg('checkpoint')
        a.s1._run_subs(sub_type='value')
        yield Msg('pause')
        a.s1._run_subs(sub_type='value')
        yield Msg('close_run')
    fresh_RE(plan(), collect)
    assert len(docs) == 3  # RunStart, EventDescriptor, one Event
    # All but one of these will be ignored. Why is one not ignored, you ask?
    # Beacuse ophyd runs subscriptions when they are (re-)subscriped.
    a.s1._run_subs(sub_type='value')
    a.s1._run_subs(sub_type='value')
    a.s1._run_subs(sub_type='value')
    a.s1._run_subs(sub_type='value')
    a.s1._run_subs(sub_type='value')
    a.s1._run_subs(sub_type='value')
    assert len(docs) == 3
    fresh_RE.resume()
    assert len(docs) == 6  # two new Events + RunStop


@requires_ophyd
def test_overlapping_read(fresh_RE):
    class DCM(Device):
        th = Cpt(Signal, value=0)
        x = Cpt(Signal, value=0)

    dcm = DCM('', name='dcm')

    def collect(name, doc):
        docs[name].append(doc)

    docs = defaultdict(list)
    fresh_RE([Msg('open_run'),
              *list(trigger_and_read([dcm.th])),
              *list(trigger_and_read([dcm])),
              Msg('close_run')], collect)
    assert len(docs['descriptor']) == 2

    docs = defaultdict(list)
    fresh_RE([Msg('open_run'),
              *list(trigger_and_read([dcm])),
              *list(trigger_and_read([dcm.th])),
              Msg('close_run')])
    assert len(docs['descriptor']) == 2

    docs = defaultdict(list)
    fresh_RE([Msg('open_run'),
              *list(trigger_and_read([dcm, dcm.th])),
              *list(trigger_and_read([dcm])),
              Msg('close_run')])
    assert len(docs['descriptor']) == 1
