import itertools
from time import time
from typing import Dict, List
from bluesky.protocols import Callback, Descriptor, Reading

from bluesky.utils import ancestry, share_ancestor, separate_devices
from bluesky.plan_stubs import trigger_and_read, declare_stream
from bluesky.preprocessors import run_decorator
from bluesky import Msg, RunEngineInterrupted
import pytest
from bluesky.tests import requires_ophyd, ophyd
from .utils import DocCollector
from unittest.mock import ANY

if ophyd:
    from ophyd import Component as Cpt, Device, Signal

    class A(Device):
        s1 = Cpt(Signal, value=0)
        s2 = Cpt(Signal, value=0)

    class B(Device):
        a1 = Cpt(A, '')
        a2 = Cpt(A, '')

    class DCM(Device):
        th = Cpt(Signal, value=0)
        x = Cpt(Signal, value=0)


class SigNew:
    def __init__(self, name: str) -> None:
        self.name = name
        self._callbacks: List[Callback] = []

    def read(self) -> Dict[str, Reading]:
        return {self.name: dict(value=0, timestamp=0)}

    def describe(self) -> Dict[str, Descriptor]:
        return {self.name: dict(source="", dtype="number", shape=[])}

    def subscribe(self, function: Callback) -> None:
        self._callbacks.append(function)

    def clear_sub(self, function: Callback) -> None:
        self._callbacks.remove(function)

    def _run_subs(self, *args, **kwargs):
        for cb in self._callbacks:
            cb({self.name: dict(value=0, timestamp=time())})


@requires_ophyd
def test_ancestry():
    b = B('', name='b')
    assert ancestry(b) == [b]
    assert ancestry(b.a1) == [b.a1, b]
    assert ancestry(b.a1.s1) == [b.a1.s1, b.a1, b]


@requires_ophyd
def test_share_ancestor():
    b1 = B('', name='b1')
    b2 = B('', name='b2')
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


@pytest.mark.parametrize('ophyd', ["v1", "v2"])
@requires_ophyd
def test_monitor(RE, ophyd):
    docs = []

    def collect(name, doc):
        docs.append(doc)

    if ophyd == "v1":
        sig = A('', name='a').s1
    else:
        sig = SigNew(name='a_s1')

    def plan():
        yield Msg('open_run')
        yield Msg('monitor', sig)
        sig._run_subs(sub_type='value')
        yield Msg('close_run')
    RE(plan(), collect)
    assert len(docs) == 4
    assert docs[2] == {
        'data': {'a_s1': 0},
        'descriptor': ANY,
        'seq_num': 1,
        'time': pytest.approx(time(), rel=0.1),
        'timestamps': {'a_s1': pytest.approx(time(), rel=0.1)},
        'filled': ANY,
        'uid': ANY
    }


@requires_ophyd
def test_monitor_with_pause_resume(RE):
    docs = []

    def collect(name, doc):
        docs.append(doc)

    a = A('', name='a')

    def plan():
        yield Msg('open_run')
        yield Msg('monitor', a.s1)
        yield Msg('checkpoint')
        a.s1._run_subs(sub_type='value')
        yield Msg('pause')
        a.s1._run_subs(sub_type='value')
        yield Msg('close_run')

    with pytest.raises(RunEngineInterrupted):
        RE(plan(), collect)
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
    RE.resume()
    assert len(docs) == 6  # two new Events + RunStop


def _make_overlapping_raising_tests(func):
    labels = ['part_v_whole',
              'whole_v_part',
              'different_obj_same_names',
              'different_names']
    if ophyd:
        dcm = DCM('', name='dcm')
        dcm2 = DCM('', name='dcm')
        dcm3 = DCM('', name='dcm3')
        dets = ((dcm.th, dcm),
                (dcm, dcm.th),
                (dcm, dcm2),
                (dcm, dcm3))
    else:
        dets = ((None, None),) * len(labels)

    return pytest.mark.parametrize('det1,det2',
                                   dets, ids=labels)(func)


@requires_ophyd
@_make_overlapping_raising_tests
def test_overlapping_raise(RE, det1, det2):

    @run_decorator()
    def test_plan(det1, det2):
        yield from declare_stream(det1, name='primary')
        yield from trigger_and_read([det1])
        yield from trigger_and_read([det2])

    with pytest.raises(RuntimeError):
        RE(test_plan(det1, det2))


def _make_overlapping_tests_2stream(func):
    labels = ['part_v_whole',
              'whole_v_part',
              'different',
              'same']
    if ophyd:
        dcm = DCM('', name='dcm')
        dcm2 = DCM('', name='dcm')
        dets = ((dcm.th, dcm),
                (dcm, dcm.th),
                (dcm, dcm2),
                (dcm, dcm))
    else:
        dets = ((None, None),) * len(labels)

    return pytest.mark.parametrize('det1,det2',
                                   dets, ids=labels)(func)


@requires_ophyd
@_make_overlapping_tests_2stream
def test_keyoverlap_2stream(RE, det1, det2):
    @run_decorator()
    def test_plan(det1, det2):
        yield from declare_stream(det1, name='primary')
        yield from trigger_and_read([det1])
        yield from declare_stream(det2, name='other')
        yield from trigger_and_read([det2], name='other')

    d = DocCollector()
    rs = RE(test_plan(det1, det2), d.insert)
    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]

    assert len(d.start) == 1
    assert len(d.descriptor[uid]) == 2


def _make_overlapping_tests_stream(func):
    labels = ['whole_whole',
              'whole_v_w+th',
              'whole_v_w+x',
              'whole_v_w+x+th']
    if ophyd:
        dcm = DCM('', name='dcm')
        dets = ((dcm, [dcm]),
                (dcm, [dcm, dcm.th]),
                (dcm, [dcm, dcm.x]),
                (dcm, [dcm, dcm.x, dcm.th])
                )
    else:
        dets = ((None, None),) * len(labels)

    return pytest.mark.parametrize('det1,det_list',
                                   dets, ids=labels)(func)


@requires_ophyd
@_make_overlapping_tests_stream
def test_overlapped_but_identical(RE, det1, det_list):
    @run_decorator()
    def test_plan(det1, det_list):
        yield from declare_stream(det1, name='primary')
        yield from trigger_and_read([det1])
        for p in itertools.permutations(det_list):
            yield from trigger_and_read(det_list)

    d = DocCollector()
    rs = RE(test_plan(det1, det_list), d.insert)
    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]

    assert len(d.start) == 1
    assert len(d.descriptor[uid]) == 1


@requires_ophyd
def test_read_clash(RE):
    dcm = DCM('', name='dcm')
    dcm2 = DCM('', name='dcm')

    with pytest.raises(ValueError):
        RE(
            [
                Msg('open_run'),
                Msg('declare_stream', None, dcm, dcm2.th, name='primary'),
                *trigger_and_read([dcm, dcm2.th]),
                Msg('close_run')
            ]
        )

    with pytest.raises(ValueError):
        RE(
            [
                Msg('open_run'),
                Msg('declare_stream', None, dcm, dcm2, name='primary'),
                *trigger_and_read([dcm, dcm2]),
                Msg('close_run')
            ]
        )

    with pytest.raises(ValueError):
        RE(
            [
                Msg('open_run'),
                Msg('declare_stream', None, dcm.th, dcm2.th, name='primary'),
                *trigger_and_read([dcm.th, dcm2.th]),
                Msg('close_run')
            ]
        )
