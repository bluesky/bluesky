from collections import defaultdict
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
    unstage,
    subscribe,
    unsubscribe,
    install_suspender,
    remove_suspender,
    open_run,
    close_run,
    wait_for,
    mv,
    mvr,
    trigger_and_read,
    stop,
    repeater,
    caching_repeater,
    repeat,
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

from bluesky.utils import all_safe_rewind

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
     (wait, (), {}, [Msg('wait', None, group=None)]),
     (wait, ('A',), {}, [Msg('wait', None, group='A')]),
     (checkpoint, (), {}, [Msg('checkpoint')]),
     (clear_checkpoint, (), {}, [Msg('clear_checkpoint')]),
     (pause, (), {}, [Msg('pause', None, defer=False)]),
     (deferred_pause, (), {}, [Msg('pause', None, defer=True)]),
     (kickoff, ('foo',), {}, [Msg('kickoff', 'foo', group=None)]),
     (kickoff, ('foo',), {'custom': 5}, [Msg('kickoff', 'foo',
                                             group=None, custom=5)]),
     (collect, ('foo',), {}, [Msg('collect', 'foo',
                                  stream=False, return_payload=True)]),
     (configure, ('det', 1), {'a': 2}, [Msg('configure', 'det', 1, a=2)]),
     (stage, ('det',), {}, [Msg('stage', 'det')]),
     (unstage, ('det',), {}, [Msg('unstage', 'det')]),
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

    def collect(name, doc):
        if name == 'descriptor':
            collector.append(doc)

    RE([Msg('open_run'),
        Msg('monitor', det, name=det.name),
        Msg('unmonitor', det),
        Msg('close_run')], collect)

    descriptor, = collector
    assert descriptor['object_keys'] == {det.name: list(det.describe().keys())}
    assert descriptor['data_keys'] == det.describe()
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

    expected = [Msg('stage', det1), Msg('read', det1), Msg('read', det1),
                Msg('stage', det2), Msg('read', det2), Msg('unstage', det2),
                Msg('unstage', det1)]

    assert processed_plan == expected


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
    assert msgs == expected

    msgs = list(trigger_and_read([det], 'custom'))
    expected = [Msg('trigger', det), Msg('wait'), Msg('create', name='custom'),
                Msg('read', det), Msg('save')]
    for msg in msgs:
        msg.kwargs.pop('group', None)
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
