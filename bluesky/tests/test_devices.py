from collections import defaultdict

from bluesky.utils import ancestry, share_ancestor, separate_devices
from bluesky.plans import trigger_and_read
from bluesky import Msg
import pytest
from bluesky.tests import requires_ophyd, ophyd

if ophyd:
    from ophyd import Component as Cpt, Device, Signal
    ophyd.setup_ophyd()

    class A(Device):
        s1 = Cpt(Signal, value=0)
        s2 = Cpt(Signal, value=0)

    class B(Device):
        a1 = Cpt(A, '')
        a2 = Cpt(A, '')

    class DCM(Device):
        th = Cpt(Signal, value=0)
        x = Cpt(Signal, value=0)


@requires_ophyd
def test_ancestry():
    b = B('', name='b')
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
    b1 = B('', name='b1')
    b2 = B('', name='b2')
    a = A('', name='a')
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

    dcm = DCM('', name='dcm')
    dcm2 = DCM('', name='dcm')

    def collect(name, doc):
        docs[name].append(doc)

    docs = defaultdict(list)
    fresh_RE(([Msg('open_run')] +
              list(trigger_and_read([dcm.th])) +
              list(trigger_and_read([dcm])) +
              [Msg('close_run')]), collect)
    assert len(docs['descriptor']) == 2

    docs = defaultdict(list)
    fresh_RE(([Msg('open_run')] +
              list(trigger_and_read([dcm])) +
              list(trigger_and_read([dcm.th])) +
              [Msg('close_run')]), collect)
    assert len(docs['descriptor']) == 2

    docs = defaultdict(list)
    fresh_RE(([Msg('open_run')] +
              list(trigger_and_read([dcm])) +
              list(trigger_and_read([dcm2])) +
              [Msg('close_run')]), collect)
    assert len(docs['descriptor']) == 2

    docs = defaultdict(list)
    fresh_RE(([Msg('open_run')] +
              list(trigger_and_read([dcm, dcm.th])) +
              list(trigger_and_read([dcm])) +
              [Msg('close_run')]), collect)
    assert len(docs['descriptor']) == 1

    docs = defaultdict(list)
    fresh_RE(([Msg('open_run')] +
              list(trigger_and_read([dcm.th, dcm.x, dcm])) +
              list(trigger_and_read([dcm])) +
              [Msg('close_run')]), collect)
    assert len(docs['descriptor']) == 1

    docs = defaultdict(list)
    fresh_RE(([Msg('open_run')] +
              list(trigger_and_read([dcm, dcm.th, dcm.x])) +
              list(trigger_and_read([dcm])) +
              [Msg('close_run')]), collect)
    assert len(docs['descriptor']) == 1


@requires_ophyd
def test_read_clash(fresh_RE):
    dcm = DCM('', name='dcm')
    dcm2 = DCM('', name='dcm')

    with pytest.raises(ValueError):
        fresh_RE(([Msg('open_run')] +
                  list(trigger_and_read([dcm, dcm2.th])) +
                  [Msg('close_run')]))

    with pytest.raises(ValueError):
        fresh_RE(([Msg('open_run')] +
                  list(trigger_and_read([dcm, dcm2])) +
                  [Msg('close_run')]))

    with pytest.raises(ValueError):
        fresh_RE(([Msg('open_run')] +
                  list(trigger_and_read([dcm.th, dcm2.th])) +
                  [Msg('close_run')]))
