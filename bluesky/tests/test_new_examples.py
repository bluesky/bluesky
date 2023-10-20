import asyncio
from collections import defaultdict
import time as ttime
from typing import Dict, List
import pytest
from bluesky import Msg, RunEngineInterrupted
from bluesky.plan_stubs import (
    create,
    save,
    drop,
    read,
    monitor,
    unmonitor,
    null,
    abs_set,
    rel_set,
    trigger,
    sleep,
    wait,
    checkpoint,
    clear_checkpoint,
    pause,
    deferred_pause,
    kickoff,
    collect,
    configure,
    stage,
    stage_all,
    unstage,
    unstage_all,
    subscribe,
    unsubscribe,
    install_suspender,
    remove_suspender,
    open_run,
    close_run,
    wait_for,
    rd,
    mv,
    mvr,
    trigger_and_read,
    stop,
    repeater,
    caching_repeater,
    repeat,
    one_shot,
    one_1d_step,
    one_nd_step)
from bluesky.preprocessors import (
    finalize_wrapper,
    fly_during_wrapper,
    reset_positions_wrapper,
    monitor_during_wrapper,
    lazily_stage_wrapper,
    relative_set_wrapper,
    subs_wrapper,
    suspend_wrapper,
    fly_during_decorator,
    subs_decorator,
    monitor_during_decorator,
    inject_md_wrapper,
    finalize_decorator,
    configure_count_time_wrapper)

from bluesky.plans import count, scan, rel_scan, inner_product_scan

import bluesky.plans as bp
from bluesky.protocols import (
    Descriptor, Locatable, Location, Readable,
    Reading, Status
)

from bluesky.utils import all_safe_rewind, IllegalMessageSequence

import threading


@pytest.mark.parametrize(
    'plan,plan_args,plan_kwargs,msgs',
    [(create, (), {}, [Msg('create', name='primary')]),
     (create, ('custom_name',), {}, [Msg('create', name='custom_name')]),
     (save, (), {}, [Msg('save')]),
     (drop, (), {}, [Msg('drop')]),
     (read, ('det',), {}, [Msg('read', 'det')]),
     (monitor, ('foo',), {}, [Msg('monitor', 'foo', name=None)]),
     (monitor, ('foo',), {'name': 'c'}, [Msg('monitor', 'foo', name='c')]),
     (unmonitor, ('foo',), {}, [Msg('unmonitor', 'foo')]),
     (null, (), {}, [Msg('null')]),
     (stop, ('foo',), {}, [Msg('stop', 'foo')]),
     (abs_set, ('det', 5), {}, [Msg('set', 'det', 5, group=None)]),
     (abs_set, ('det', 5), {'group': 'A'}, [Msg('set', 'det', 5, group='A')]),
     (abs_set, ('det', 5), {'group': 'A', 'wait': True},
      [Msg('set', 'det', 5, group='A'), Msg('wait', None, group='A')]),
     (rel_set, ('det', 5), {}, [Msg('read', 'det'),
                                Msg('set', 'det', 5, group=None)]),
     (rel_set, ('det', 5), {'group': 'A'}, [Msg('read', 'det'),
                                            Msg('set', 'det', 5, group='A')]),
     (rel_set, ('det', 5), {'group': 'A', 'wait': True},
      [Msg('read', 'det'), Msg('set', 'det', 5, group='A'),
       Msg('wait', None, group='A')]),
     (trigger, ('det',), {}, [Msg('trigger', 'det', group=None)]),
     (trigger, ('det',), {'group': 'A'}, [Msg('trigger', 'det', group='A')]),
     (sleep, (2,), {}, [Msg('sleep', None, 2)]),
     (wait, (), {}, [Msg('wait', None, group=None, timeout=None)]),
     (wait, ('A',), {}, [Msg('wait', None, group='A', timeout=None)]),
     (checkpoint, (), {}, [Msg('checkpoint')]),
     (clear_checkpoint, (), {}, [Msg('clear_checkpoint')]),
     (pause, (), {}, [Msg('pause', None, defer=False)]),
     (deferred_pause, (), {}, [Msg('pause', None, defer=True)]),
     (kickoff, ('foo',), {}, [Msg('kickoff', 'foo', group=None)]),
     (kickoff, ('foo',), {'custom': 5}, [Msg('kickoff', 'foo',
                                             group=None, custom=5)]),
     (collect, ('foo',), {}, [Msg('collect', 'foo',
                                  stream=False, return_payload=True, name=None)]),
     (configure, ('det', 1), {'a': 2}, [Msg('configure', 'det', 1, a=2)]),
     (stage, ('det',), {}, [Msg('stage', 'det', group=None)]),
     (stage, ('det',), {"group": "A"}, [Msg('stage', 'det', group="A")]),
     (unstage, ('det',), {}, [Msg('unstage', 'det', group=None)]),
     (unstage, ('det',), {"group": "A"}, [Msg('unstage', 'det', group="A")]),
     (subscribe, ('all', 'func_placeholder'), {}, [Msg('subscribe', None,
                                                       'func_placeholder',
                                                       'all')]),
     (unsubscribe, (1,), {}, [Msg('unsubscribe', None, token=1)]),
     (install_suspender, (1,), {}, [Msg('install_suspender', None, 1)]),
     (remove_suspender, (1,), {}, [Msg('remove_suspender', None, 1)]),
     (open_run, (), {}, [Msg('open_run')]),
     (open_run, (), {'md': {'a': 1}}, [Msg('open_run', a=1)]),
     (close_run, (), {}, [Msg('close_run', reason=None, exit_status=None)]),
     (wait_for, (['fut1', 'fut2'],), {}, [Msg('wait_for', None,
                                              ['fut1', 'fut2'])]),
     ]
)
def test_stub_plans(plan, plan_args, plan_kwargs, msgs, hw):
    # de-reference
    plan_args = tuple(getattr(hw, v, v) if isinstance(v, str) else v
                      for v in plan_args)
    plan_kwargs = {k: getattr(hw, v, v) if isinstance(v, str) else v
                   for k, v in plan_kwargs.items()}
    msgs = [Msg(m.command,
                getattr(hw, m.obj, m.obj) if isinstance(m.obj, str) else m.obj,
                *m.args,
                **m.kwargs) for m in msgs]
    assert list(plan(*plan_args, **plan_kwargs)) == msgs


def test_mv(hw):
    # special-case mv because the group is not configurable
    # move motors first to ensure that movement is absolute, not relative
    actual = list(mv(hw.motor1, 1, hw.motor2, 2))
    strip_group(actual)
    for msg in actual[:2]:
        msg.command == 'set'
    assert set([msg.obj for msg in actual[:2]]) == set([hw.motor1, hw.motor2])
    assert actual[2] == Msg('wait', None)


def test_mv_with_timeout(hw):
    # special-case mv because the group is not configurable
    # move motors first to ensure that movement is absolute, not relative
    actual = list(mv(hw.motor1, 1, hw.motor2, 2, timeout=42))
    for msg in actual[:2]:
        msg.command == 'set'
        msg.kwargs['timeout'] == 42


def test_mvr(RE, hw):
    # special-case mv because the group is not configurable
    # move motors first to ensure that movement is relative, not absolute
    hw.motor1.set(10)
    hw.motor2.set(10)
    actual = []
    RE.msg_hook = lambda msg: actual.append(msg)
    RE(mvr(hw.motor1, 1, hw.motor2, 2))
    actual = list(mv(hw.motor1, 1, hw.motor2, 2))
    strip_group(actual)
    for msg in actual[:2]:
        msg.command == 'set'
    assert set([msg.obj for msg in actual[:2]]) == set([hw.motor1, hw.motor2])
    assert actual[2] == Msg('wait', None)


def test_locatable_message_multiple_objects(RE, hw):
    class AsyncLocatable(Locatable):
        value = 1.0

        def set(self, value) -> Status:
            self.position = value + 0.1
            # We don't actually need a motor, but use one just to give us a Status
            return hw.motor.set(value)

        async def locate(self) -> Location:
            # Grab the original value
            value = AsyncLocatable.value
            # Let any other coros get a look in
            await asyncio.sleep(0)
            # Increment the next value and return the original value
            AsyncLocatable.value += 1
            return dict(setpoint=value, readback=value)

    rds = []
    one = AsyncLocatable()
    two = AsyncLocatable()

    def multi_rd():
        rds.append((yield Msg("locate", one, squeeze=False)))
        rds.append((yield Msg("locate", two)))
        rds.append((yield Msg("locate", one, two)))
        rds.append((yield Msg("locate", one)))

    RE(multi_rd())
    assert rds == [
        [dict(setpoint=1.0, readback=1.0)],
        dict(setpoint=2.0, readback=2.0),
        # Check they happened at the same time, so have the same value
        [dict(setpoint=3.0, readback=3.0), dict(setpoint=3.0, readback=3.0)],
        # And now it will skip one as they both incremented above
        dict(setpoint=5.0, readback=5.0),
    ]


def test_rd_locatable(RE, hw):
    class Jittery(Readable, Locatable):
        def describe(self) -> Dict[str, Descriptor]:
            return dict(x=dict(source="dummy", dtype="number", shape=[]))

        def read(self) -> Dict[str, Reading]:
            return dict(x=dict(value=1.2, timestamp=0.0))

        def locate(self) -> Location:
            return dict(setpoint=1.0, readback=1.1)

        def set(self, value) -> Status:
            self.position = value + 0.1
            # We don't actually need a motor, but use one just to give us a Status
            return hw.motor.set(value)

        name = ""

    jittery = Jittery()
    rds = []

    def store_rd():
        value = yield from rd(jittery)
        rds.append(value)

    RE(store_rd())
    assert rds == [1.1]


def test_mvr_with_location(RE, hw):
    class AlmostThereMotor:
        # This is the readback
        position = 10.1
        parent = None
        name = "Almost"

        def set(self, value) -> Status:
            self.position = value + 0.1
            # We don't actually need a motor, but use one just to give us a Status
            return hw.motor.set(value)

        async def locate(self) -> Location:
            await asyncio.sleep(0.1)
            return dict(setpoint=self.position-0.1, readback=self.position)

    m = AlmostThereMotor()
    assert isinstance(m, Locatable)
    actual = []
    RE.msg_hook = lambda msg: actual.append(msg)
    RE(mvr(m, 1))
    RE(mvr(m, 2))
    assert [x.command for x in actual] == ["locate", "set", "wait"] * 2
    assert [x.obj for x in actual] == [m, m, None, m, m, None]
    assert actual[1].args == (11,)
    assert actual[4].args == (13,)


def test_mvr_with_timeout(hw):
    # special-case mv because the group is not configurable
    # move motors first to ensure that movement is absolute, not relative
    actual = list(mvr(hw.motor1, 1, hw.motor2, 2, timeout=42))
    for msg in actual[:2]:
        msg.command == 'set'
        msg.kwargs['timeout'] == 42


def strip_group(plan):
    for msg in plan:
        msg.kwargs.pop('group', None)


def test_monitor_during_wrapper(hw):
    det = hw.det

    def plan():
        # can't use 2 * [Msg('open_run'), Msg('null'), Msg('close_run')]
        # because plan_mutator sees the same ids twice and skips them
        yield from [Msg('open_run'), Msg('null'), Msg('close_run'),
                    Msg('open_run'), Msg('null'), Msg('close_run')]

    processed_plan = list(monitor_during_wrapper(plan(), [det]))
    expected = 2 * [Msg('open_run'),
                    # inserted
                    Msg('monitor', det, name=(det.name + '_monitor')),
                    Msg('null'),
                    Msg('unmonitor', det),  # inserted
                    Msg('close_run')]

    strip_group(processed_plan)
    assert processed_plan == expected

    processed_plan = list(monitor_during_decorator([det])(plan)())
    strip_group(processed_plan)
    assert processed_plan == expected


def test_descriptor_layout_from_monitor(RE, hw):
    collector = []
    det = hw.rand
    det.start_simulation()

    def collect(name, doc):
        if name == 'descriptor':
            collector.append(doc)

    RE([Msg('open_run'),
        Msg('monitor', det, name=det.name),
        Msg('unmonitor', det),
        Msg('close_run')], collect)

    descriptor, = collector
    assert descriptor['object_keys'] == {det.name: list(det.describe().keys())}
    assert descriptor['data_keys'] == {
        k: {**v, "object_name": det.name} for k, v in det.describe().items()
    }
    conf = descriptor['configuration'][det.name]
    assert conf['data_keys'] == det.describe_configuration()
    vals = {key: val['value'] for key, val in det.read_configuration().items()}
    timestamps = {key: val['timestamp']
                  for key, val in det.read_configuration().items()}
    assert conf['data'] == vals
    assert conf['timestamps'].keys() == timestamps.keys()
    for val in conf['timestamps'].values():
        assert type(val) is float  # can't check actual value; time has passed


def test_fly_during():
    def plan():
        # can't use 2 * [Msg('open_run'), Msg('null'), Msg('close_run')]
        # because plan_mutator sees the same ids twice and skips them
        yield from [Msg('open_run'), Msg('null'), Msg('close_run'),
                    Msg('open_run'), Msg('null'), Msg('close_run')]

    processed_plan = list(fly_during_wrapper(plan(), ['foo']))
    expected = 2 * [Msg('open_run'),
                    Msg('kickoff', 'foo'), Msg('wait'),  # inserted
                    Msg('null'),
                    Msg('complete', 'foo'), Msg('wait'),  # inserted
                    Msg('collect', 'foo'),  # inserted
                    Msg('close_run')]

    strip_group(processed_plan)
    assert processed_plan == expected

    processed_plan = list(fly_during_decorator(['foo'])(plan)())
    strip_group(processed_plan)
    assert processed_plan == expected


def test_lazily_stage(hw):
    det1, det2 = hw.det1, hw.det2

    def plan():
        yield from [Msg('read', det1), Msg('read', det1), Msg('read', det2)]

    processed_plan = list(lazily_stage_wrapper(plan()))

    expected_plan: List[Msg] = [
        Msg('stage', det1), Msg('read', det1), Msg('read', det1),
        Msg('stage', det2), Msg('read', det2), Msg('unstage', det2),
        Msg('unstage', det1)
    ]

    # Prevent issue with unstage_all creating a randomly assigned group
    assert len(processed_plan) == len(expected_plan)
    for i in range(len(expected_plan)):
        expected, actual = expected_plan[i], processed_plan[i]
        assert actual.command == expected.command
        assert actual.obj == expected.obj
    groups = {msg.kwargs["group"] for msg in processed_plan if msg.command == "unstage"}
    assert len(groups) == 1  # All unstage messages are in the same group


def test_subs():

    def cb(name, doc):
        pass

    def plan(*args, **kwargs):
        # check that args to plan are passed through
        yield from [Msg('null', None, *args, **kwargs)]

    processed_plan = list(subs_wrapper(plan('test_arg', test_kwarg='val'),
                                       {'all': cb}))

    expected = [Msg('subscribe', None, cb, 'all'),
                Msg('null', None, 'test_arg', test_kwarg='val'),
                Msg('unsubscribe', token=None)]

    assert processed_plan == expected

    processed_plan = list(subs_decorator({'all': cb})(plan)('test_arg',
                                                            test_kwarg='val'))
    assert processed_plan == expected


def test_suspend():
    plan = [Msg('null')]

    processed_plan = list(suspend_wrapper(plan, 1))

    expected = [Msg('install_suspender', None, 1),
                Msg('null'),
                Msg('remove_suspender', None, 1)]

    assert processed_plan == expected


def test_md():
    def plan():
        yield from open_run(md={'a': 1})

    processed_plan = list(inject_md_wrapper(plan(), {'b': 2}))

    expected = [Msg('open_run', None, a=1, b=2)]

    assert processed_plan == expected


def test_finalize(hw):
    det = hw.det

    def plan():
        yield from [Msg('null')]

    def cleanup_plan():
        yield from [Msg('read', det)]

    # wrapper accepts list
    processed_plan = list(finalize_wrapper(plan(), [Msg('read', det)]))
    expected = [Msg('null'), Msg('read', det)]
    assert processed_plan == expected

    # or func that returns list
    def plan():
        yield from [Msg('null')]

    processed_plan = list(finalize_wrapper(plan(), lambda: [Msg('read', det)]))
    expected = [Msg('null'), Msg('read', det)]
    assert processed_plan == expected

    # or generator instance
    def plan():
        yield from [Msg('null')]

    processed_plan = list(finalize_wrapper(plan(), cleanup_plan()))
    expected = [Msg('null'), Msg('read', det)]
    assert processed_plan == expected

    # or generator func
    def plan():
        yield from [Msg('null')]

    processed_plan = list(finalize_wrapper(plan(), cleanup_plan))
    expected = [Msg('null'), Msg('read', det)]
    assert processed_plan == expected

    # decorator accepts generator func
    processed_plan = list(finalize_decorator(cleanup_plan)(plan)())
    expected = [Msg('null'), Msg('read', det)]
    assert processed_plan == expected

    # or func that returns list
    processed_plan = list(finalize_decorator(
        lambda: [Msg('read', det)])(plan)())
    expected = [Msg('null'), Msg('read', det)]
    assert processed_plan == expected

    # decorator does NOT accept list
    with pytest.raises(TypeError):
        list(finalize_decorator([Msg('read', det)])(plan)())

    # nor generator instance
    with pytest.raises(TypeError):
        list(finalize_decorator(cleanup_plan())(plan)())


def test_finalize_runs_after_error(RE, hw):
    det = hw.det

    def plan():
        yield from [Msg('null')]
        raise Exception

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    RE.msg_hook = accumulator
    try:
        RE(finalize_wrapper(plan(), [Msg('read', det)]))
    except Exception:
        pass  # swallow the Exception; we are interested in msgs below

    expected = [Msg('null'), Msg('read', det)]

    assert msgs == expected


def test_reset_positions(RE, hw):
    motor = hw.motor
    motor.set(5)

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    RE(reset_positions_wrapper(plan()))

    expected = [Msg('set', motor, 8), Msg('set', motor, 5), Msg('wait')]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_reset_positions_no_position_attr(RE, hw):
    motor = hw.motor_no_pos
    motor.set(5)

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    RE(reset_positions_wrapper(plan()))

    expected = [Msg('read', motor),
                Msg('set', motor, 8), Msg('set', motor, 5), Msg('wait')]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_relative_set(RE, hw):
    motor = hw.motor
    motor.set(5)

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    RE(relative_set_wrapper(plan()))

    expected = [Msg('set', motor, 13)]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_relative_set_no_position_attr(RE, hw):
    motor = hw.motor_no_pos
    motor.set(5)

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    RE(relative_set_wrapper(plan()))

    expected = [Msg('read', motor),
                Msg('set', motor, 13)]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_configure_count_time(RE, hw):
    det = hw.det_with_count_time
    det.count_time.put(3)

    def plan():
        yield from (m for m in [Msg('read', det)])

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    RE.msg_hook = accumulator

    RE(configure_count_time_wrapper(plan(), 7))

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
    with pytest.warns(UserWarning):
        actual = list(caching_repeater(3, plan_instance))
    assert actual == 3 * [1, 2, 3]

    plan_instance = plan(1, 2, 3)
    p = caching_repeater(None, plan_instance)
    with pytest.warns(UserWarning):
        assert next(p) == 1
    assert next(p) == 2
    assert next(p) == 3
    assert next(p) == 1


def test_repeat(RE):
    # Check if lists and callables both work
    messages = [1, 2, 3]

    def plan():
        yield from messages

    num = 3
    expected = [Msg('checkpoint'), 1, 2, 3] * num
    assert list(repeat(plan, num=num)) == expected


def test_repeat_using_RE(RE):
    def plan():
        yield Msg('open_run')
        yield Msg('close_run')
    RE(repeat(plan, 2))


def test_trigger_and_read(hw):
    det = hw.det
    msgs = list(trigger_and_read([det]))
    expected = [Msg('trigger', det), Msg('wait'),
                Msg('create', name='primary'), Msg('read', det), Msg('save')]
    for msg in msgs:
        msg.kwargs.pop('group', None)
        msg.kwargs.pop('timeout', None)
    assert msgs == expected

    msgs = list(trigger_and_read([det], 'custom'))
    expected = [Msg('trigger', det), Msg('wait'), Msg('create', name='custom'),
                Msg('read', det), Msg('save')]
    for msg in msgs:
        msg.kwargs.pop('group', None)
        msg.kwargs.pop('timeout', None)
    assert msgs == expected


def test_count_delay_argument(hw):
    # num=7 but delay only provides 5 entries
    with pytest.raises(ValueError):
        # count raises ValueError when delay generator is expired
        list(count([hw.det], num=7, delay=(2**i for i in range(5))))

    # num=6 with 5 delays between should product 6 readings
    msgs = count([hw.det], num=6, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 6

    # num=5 with 5 delays should produce 5 readings
    msgs = count([hw.det], num=5, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 5

    # num=4 with 5 delays should produce 4 readings
    msgs = count([hw.det], num=4, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 4

    # num=None with 5 delays should produce 6 readings
    msgs = count([hw.det], num=None, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 6


def test_plan_md(RE, hw):
    mutable = []
    md = {'color': 'red'}

    def collector(name, doc):
        mutable.append(doc)

    # test genereator
    mutable.clear()
    RE(count([hw.det], md=md), collector)
    assert 'color' in mutable[0]

    # test Plan with explicit __init__
    mutable.clear()
    RE(bp.count([hw.det], md=md), collector)
    assert 'color' in mutable[0]

    # test Plan with implicit __init__ (created via metaclasss)
    mutable.clear()
    RE(bp.scan([hw.det], hw.motor, 1, 2, 2, md=md), collector)
    assert 'color' in mutable[0]


def test_infinite_count(RE, hw):
    threading.Timer(1, RE.stop).start()
    docs = defaultdict(list)

    def collector(name, doc):
        docs[name].append(doc)

    with pytest.raises(RunEngineInterrupted):
        RE(count([hw.det], num=None), collector)

    assert len(docs['start']) == 1
    assert len(docs['stop']) == 1
    assert len(docs['descriptor']) == 1
    assert len(docs['event']) > 0


def test_no_rewind_device(hw):
    hw.det.rewindable = hw.bool_sig

    assert not all_safe_rewind([hw.det])


def test_monitor(RE, hw):
    from ophyd.sim import SynSignal
    signal = SynSignal(name='signal')
    signal.put(0.0)
    RE(monitor_during_wrapper(count([hw.det], 5), [signal]))


def test_per_step(RE, hw):
    # Check default behavior, using one motor and then two.
    RE(scan([hw.det], hw.motor, -1, 1, 3, per_step=one_nd_step))
    RE(scan([hw.det],
            hw.motor, -1, 1,
            hw.motor2, -1, 1,
            3,
            per_step=one_nd_step))
    RE(inner_product_scan([hw.det], 3, hw.motor, -1, 1, per_step=one_nd_step))
    RE(inner_product_scan([hw.det],
                          3,
                          hw.motor, -1, 1,
                          hw.motor2, -1, 1,
                          per_step=one_nd_step))

    # Check that scan still accepts old one_1d_step signature:
    RE(scan([hw.det], hw.motor, -1, 1, 3, per_step=one_1d_step))
    RE(rel_scan([hw.det], hw.motor, -1, 1, 3, per_step=one_1d_step))

    # Test that various error paths include a useful error message identifying
    # that the problem is with 'per_step':

    # You can't usage one_1d_step signature with more than one motor.
    with pytest.raises(TypeError) as excinfo:
        RE(scan([hw.det],
                hw.motor, -1, 1,
                hw.motor2, -1, 1,
                3,
                per_step=one_1d_step))
    assert excinfo.match("Signature of per_step assumes 1D trajectory")

    # The signature must be either like one_1d_step or one_nd_step:
    def bad_sig(detectors, mtr, step):
        ...

    with pytest.raises(TypeError) as excinfo:
        RE(scan([hw.det], hw.motor, -1, 1, 3, per_step=bad_sig))
    assert excinfo.match("per_step must be a callable with the signature")


def test_stage_all_and_unstage_all(RE):

    from ophyd import StatusBase

    staged = {}
    unstaged = {}

    # Test support for old and new style devices
    class OldStyleDummy:
        def __init__(self, name: str) -> None:
            self.name: str = name

        def stage(self):
            staged[self.name] = True
            ttime.sleep(0.5)
            return [self]

        def unstage(self):
            unstaged[self.name] = True
            ttime.sleep(0.5)
            return [self]

    class NewStyleDummy:
        def __init__(self, name: str) -> None:
            self.name: str = name

        def _callback(self, d: dict, st: Status):
            d[self.name] = True
            st.set_finished()

        def stage(self) -> Status:
            st = StatusBase()
            threading.Timer(0.5, self._callback, args=[staged, st]).start()
            return st

        def unstage(self) -> Status:
            st = StatusBase()
            threading.Timer(0.5, self._callback, args=[unstaged, st]).start()
            return st

    olddummy1 = OldStyleDummy("o1")
    olddummy2 = OldStyleDummy("o2")
    newdummy1 = NewStyleDummy("n1")
    newdummy2 = NewStyleDummy("n2")

    def plan():
        yield from stage_all(olddummy1, olddummy2, newdummy1, newdummy2)
        assert list(staged.keys()) == ["o1", "o2", "n1", "n2"]

        yield from unstage_all(olddummy1, olddummy2, newdummy1, newdummy2)
        assert list(unstaged.keys()) == ["o1", "o2", "n1", "n2"]

    start = ttime.monotonic()
    RE(plan())
    stop = ttime.monotonic()

    assert 3 < stop - start < 4


def test_old_style_wait(RE):

    class OldStyleDummy:
        def __init__(self, name: str) -> None:
            self.name: str = name

        def stage(self):
            ttime.sleep(0.5)
            return [self]

        def unstage(self):
            ttime.sleep(0.5)
            return [self]

    dummy = OldStyleDummy("o1")

    def stage_plan():
        yield from stage(dummy, wait=False)

    def unstage_plan():
        yield from unstage(dummy, wait=False)

    with pytest.raises(RuntimeError):
        RE(stage_plan())

    with pytest.raises(RuntimeError):
        RE(unstage_plan())


def test_custom_stream_name(RE, hw):

    def new_trigger_and_read(devices):
        return (yield from trigger_and_read(devices, name='secondary'))

    def new_per_step(detectors, motor, step):
        return (yield from one_1d_step(detectors, motor,
                                       step, take_reading=new_trigger_and_read))

    def new_per_shot(detectors):
        return (yield from one_shot(detectors, take_reading=new_trigger_and_read))

    RE(scan([hw.det], hw.motor, -1, 1, 3, per_step=new_per_step))
    RE(count([hw.det], 3, per_shot=new_per_shot))

    RE._require_stream_declaration = True
    with pytest.raises(IllegalMessageSequence):
        RE(scan([hw.det], hw.motor, -1, 1, 3, per_step=new_per_step))

    with pytest.raises(IllegalMessageSequence):
        RE(count([hw.det], 3, per_shot=new_per_shot))

    with pytest.raises(IllegalMessageSequence):
        RE(count([hw.det], 3, per_shot=one_shot))
