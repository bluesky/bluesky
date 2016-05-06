import copy
from collections import deque
import pytest
from bluesky import Msg
from bluesky.examples import det, det1, det2, Mover
from bluesky.plans import (create, save, read, monitor, unmonitor, null,
                           abs_set, rel_set, trigger, sleep, wait, checkpoint,
                           clear_checkpoint, pause, deferred_pause, kickoff,
                           collect, configure, stage, unstage, subscribe,
                           unsubscribe, open_run, close_run, wait_for,
                           subs_context, run_context, event_context,
                           baseline_context, monitor_context,
                           stage_context, planify, finalize, fly_during,
                           lazily_stage, relative_set, reset_positions,
                           configure_count_time, trigger_and_read,
                           repeater, caching_repeater)


class Status:
    "a simple Status object that is always immediately done"
    def __init__(self):
        self._cb = None
        self.done = True
        self.success = True

    @property
    def finished_cb(self):
        return self._cb

    @finished_cb.setter
    def finished_cb(self, cb):
        cb()
        self._cb = cb

class DummyMover:
    def __init__(self, name):
        self._value = 0
        self.name = name

    def describe(self):
        return {self.name: {}}

    def set(self, value):
        self._value = value
        return Status()

    def read_configuration(self):
        return {}

    def describe_configuration(self):
        return {}

    def read(self):
        return {self.name: {'value': self._value, 'timestamp': 0}}



def cb(name, doc):
    pass

@pytest.mark.parametrize(
    'plan,plan_args,plan_kwargs,msgs',
    [(create, (), {}, [Msg('create', name='primary')]),
     (create, ('custom_name',), {}, [Msg('create', name='custom_name')]),
     (save, (), {}, [Msg('save')]),
     (read, (det,), {}, [Msg('read', det)]),
     (monitor, ('foo',), {}, [Msg('monitor', 'foo', name=None)]), 
     (monitor, ('foo',), {'name': 'c'}, [Msg('monitor', 'foo', name='c')]), 
     (unmonitor, ('foo',), {}, [Msg('unmonitor', 'foo')]),
     (null, (), {}, [Msg('null')]),
     (abs_set, (det, 5), {}, [Msg('set', det, 5, group=None)]),
     (abs_set, (det, 5), {'group': 'A'}, [Msg('set', det, 5, group='A')]),
     (abs_set, (det, 5), {'group': 'A', 'wait': True},
      [Msg('set', det, 5, group='A'), Msg('wait', None, group='A')]),
     (rel_set, (det, 5), {}, [Msg('read', det),
                              Msg('set', det, 5, group=None)]),
     (rel_set, (det, 5), {'group': 'A'}, [Msg('read', det),
                                          Msg('set', det, 5, group='A')]),
     (rel_set, (det, 5), {'group': 'A', 'wait': True},
      [Msg('read', det), Msg('set', det, 5, group='A'),
       Msg('wait', None, group='A')]),
     (trigger, (det,), {}, [Msg('trigger', det, group=None)]),
     (trigger, (det,), {'group': 'A'}, [Msg('trigger', det, group='A')]),
     (sleep, (2,), {}, [Msg('sleep', None, 2)]),
     (wait, (), {}, [Msg('wait', None, group=None)]),
     (wait, ('A',), {}, [Msg('wait', None, group='A')]),
     (checkpoint, (), {}, [Msg('checkpoint')]),
     (clear_checkpoint, (), {}, [Msg('clear_checkpoint')]),
     (pause, (), {}, [Msg('pause', None, defer=False)]),
     (deferred_pause, (), {}, [Msg('pause', None, defer=True)]),
     (kickoff, ('foo',), {}, [Msg('kickoff', 'foo', name=None, group=None)]),
     (kickoff, ('foo',), {'custom': 5}, [Msg('kickoff', 'foo', name=None,
                                             group=None, custom=5)]),
     (collect, ('foo',), {}, [Msg('collect', 'foo', stream=False)]),
     (configure, (det, 1), {'a': 2}, [Msg('configure', det, 1, a=2)]),
     (stage, (det,), {}, [Msg('stage', det)]),
     (unstage, (det,), {}, [Msg('unstage', det)]),
     (subscribe, ('all', 'func_placeholder'), {}, [Msg('subscribe', None, 'all',
                                                       'func_placeholder')]),
     (unsubscribe, (1,), {}, [Msg('unsubscribe', None, token=1)]),
     (open_run, (), {}, [Msg('open_run')]),
     (open_run, (), {'md': {'a': 1}}, [Msg('open_run', a=1)]),
     (close_run, (), {}, [Msg('close_run')]),
     (wait_for, (['fut1', 'fut2'],), {}, [Msg('wait_for', None, ['fut1', 'fut2'])]),
    ])
def test_stub_plans(plan, plan_args, plan_kwargs, msgs):
    assert list(plan(*plan_args, **plan_kwargs)) == msgs


@pytest.mark.parametrize(
    'cm,args,kwargs,before,after',
    [(baseline_context, ([det1, det2],), {},
      [Msg('trigger', det1),
       Msg('trigger', det2),
       Msg('wait'),
       Msg('create', None, name='baseline'),
       Msg('read', det1),
       Msg('read', det2),
       Msg('save')],
      [Msg('trigger', det1),
       Msg('trigger', det2),
       Msg('wait'),
       Msg('create', None, name='baseline'),
       Msg('read', det1),
       Msg('read', det2),
       Msg('save')]),
     (stage_context, ([det1, det2],), {},
      [Msg('stage', det1),
       Msg('stage', det2)],
      [Msg('unstage', det2),
       Msg('unstage', det1)]),
     (subs_context, ({'all': [cb]},), {},
      [Msg('subscribe', None, 'all', cb)],
      [Msg('unsubscribe', None, token=None)]),
     (monitor_context, (['sig'],), {},
      list(monitor('sig')),
      list(unmonitor('sig'))),
     (event_context, (), {},
      [Msg('create', name='primary')],
      [Msg('save')]),
     (run_context, (), {'md': {'a': 1}},
      [Msg('open_run', a=1)],
      [Msg('close_run')]),
    ])
def test_plan_contexts(cm, args, kwargs, before, after):
    @planify
    def plan():
        ps = deque()
        with cm(ps, *args, **kwargs):
            ps.append(['sentinel'])
        return ps

    actual_before = []
    actual_after = []
    target = actual_before
    for msg in plan():
        if msg == 'sentinel':
            target = actual_after
            continue
        msg.kwargs.pop('group', None)
        target.append(msg)
    assert actual_before == before
    assert actual_after == after


def test_fly_during():
    def plan():
        # can't use 2 * [Msg('open_run'), Msg('null'), Msg('close_run')]
        # because plan_mutator sees the same ids twice and skips them
        yield from [Msg('open_run'), Msg('null'), Msg('close_run'),
                    Msg('open_run'), Msg('null'), Msg('close_run')]

    processed_plan = list(fly_during(plan(), ['foo']))
    expected = 2 * [Msg('open_run'), Msg('kickoff', 'foo'), Msg('null'),
                    Msg('wait'), Msg('collect', 'foo'), Msg('close_run')]

    for msg in processed_plan:
        msg.kwargs.pop('group', None)

    assert processed_plan == expected


def lazily_stage():
    def plan():
        yield from [Msg('read', det1), Msg('read', det1), Msg('read', det2)]

    processed_plan = list(lazily_stage(plan()))

    expected = [Msg('stage', det1), Msg('read', det1), Msg('read', det1),
                Msg('stage', det2), Msg('read', det2)]

    assert processed_plan == expected


def test_finalize():
    def plan():
        yield from [Msg('null')]

    processed_plan = list(finalize(plan(), [Msg('read', det)]))

    expected = [Msg('null'), Msg('read', det)]

    assert processed_plan == expected


def test_finalize_runs_after_error(fresh_RE):
    def plan():
        yield from [Msg('null')]
        raise Exception

    msgs = []
        
    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator
    try:
        fresh_RE(finalize(plan(), [Msg('read', det)]))
    except Exception:
        pass # swallow the Exception; we are interested in msgs below

    expected = [Msg('null'), Msg('read', det)]

    assert msgs == expected


def test_reset_positions(fresh_RE):
    motor = Mover('a', ['a'])
    motor.set(5)

    msgs = []
        
    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    fresh_RE(reset_positions(plan()))

    expected = [Msg('set', motor, 8), Msg('set', motor, 5), Msg('wait')]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_reset_positions_no_position_attr(fresh_RE):
    motor = DummyMover('motor')
    motor.set(5)

    msgs = []
        
    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    fresh_RE(reset_positions(plan()))

    expected = [Msg('read', motor),
                Msg('set', motor, 8), Msg('set', motor, 5), Msg('wait')]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_relative_set(fresh_RE):
    motor = Mover('a', ['a'])
    motor.set(5)

    msgs = []
        
    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    fresh_RE(relative_set(plan()))

    expected = [Msg('set', motor, 13)]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_relative_set_no_position_attr(fresh_RE):
    motor = DummyMover('motor')
    motor.set(5)

    msgs = []
        
    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    fresh_RE(relative_set(plan()))

    expected = [Msg('read', motor),
                Msg('set', motor, 13)]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_configure_count_time(fresh_RE):
    class DummySignal:
        def put(self, val):
            pass

        def get(self):
            return 3

        def set(self, val):
            return Status()

    det = DummyMover('det')
    det.count_time = DummySignal()

    def plan():
        yield from (m for m in [Msg('read', det)])

    msgs = []
        
    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    fresh_RE(configure_count_time(plan(), 7))

    expected = [Msg('set', det.count_time, 7), Msg('wait'),
                Msg('read', det), Msg('set', det.count_time, 3),
                Msg('wait')]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_repeater():
    def plan(*args):
        yield from args

    actual = list(repeater(3, plan, 1, 2, 3))
    assert actual == 3 * [1, 2, 3]

    p = repeater(None, plan, 1, 2, 3)
    assert next(p) == 1
    assert next(p) == 2
    assert next(p) == 3
    assert next(p) == 1


def test_caching_repeater():
    def plan(*args):
        yield from args

    plan_instance = plan(1, 2, 3)
    actual = list(caching_repeater(3, plan_instance))
    assert actual == 3 * [1, 2, 3]

    plan_instance = plan(1, 2, 3)
    p = caching_repeater(None, plan_instance)
    assert next(p) == 1
    assert next(p) == 2
    assert next(p) == 3
    assert next(p) == 1


def test_trigger_and_read():
    msgs = list(trigger_and_read([det]))
    expected = [Msg('trigger', det), Msg('wait'),
                Msg('create', name='primary'), Msg('read', det), Msg('save')]
    for msg in msgs:
        msg.kwargs.pop('group', None)
    assert msgs == expected

    msgs = list(trigger_and_read([det], 'custom'))
    expected = [Msg('trigger', det), Msg('wait'), Msg('create', name='custom'),
                Msg('read', det), Msg('save')]
    for msg in msgs:
        msg.kwargs.pop('group', None)
    assert msgs == expected
