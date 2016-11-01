from collections import deque, defaultdict
import pytest
from bluesky import Msg
from bluesky.examples import (det, det1, det2, Mover, NullStatus, motor,
                              SynGauss, Reader)
from bluesky.plans import (create, save, read, monitor, unmonitor, null,
                           abs_set, rel_set, trigger, sleep, wait, checkpoint,
                           clear_checkpoint, pause, deferred_pause, kickoff,
                           collect, configure, stage, unstage, subscribe,
                           unsubscribe, open_run, close_run, wait_for, mv,
                           subs_context, run_context, event_context,
                           baseline_context, monitor_context,
                           stage_context, planify, finalize_wrapper,
                           fly_during_wrapper, reset_positions_wrapper,
                           monitor_during_wrapper,
                           lazily_stage_wrapper, relative_set_wrapper,
                           configure_count_time_wrapper,
                           subs_wrapper, trigger_and_read, stop,
                           repeater, caching_repeater, count, Count, Scan,
                           fly_during_decorator, subs_decorator,
                           monitor_during_decorator,
                           inject_md_wrapper, finalize_decorator)
from bluesky.utils import all_safe_rewind


class DummyMover:
    def __init__(self, name):
        self._value = 0
        self.name = name

    def describe(self):
        return {self.name: {}}

    def set(self, value):
        self._value = value
        return NullStatus()

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
     (stop, ('foo',), {}, [Msg('stop', 'foo')]),
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
     (kickoff, ('foo',), {}, [Msg('kickoff', 'foo', group=None)]),
     (kickoff, ('foo',), {'custom': 5}, [Msg('kickoff', 'foo',
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


def test_mv():
    # special-case mv because the group is not configurable
    actual = list(mv(det1, 1, det2, 2))
    expected = [Msg('set', det1, 1, group=None),
                Msg('set', det2, 2, group=None),
                Msg('wait', None, group=None)]
    strip_group(actual)
    strip_group(expected)
    assert actual == expected


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


def strip_group(plan):
    for msg in plan:
        msg.kwargs.pop('group', None)


def test_monitor_during_wrapper():
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


def test_descriptor_layout_from_monitor(fresh_RE):
    collector = []
    det = Reader('det', {k: lambda: i for i, k in enumerate('abcd')},
                 read_attrs=list('ab'), conf_attrs=list('cd'))

    def collect(name, doc):
        if name == 'descriptor':
            collector.append(doc)

    fresh_RE([Msg('open_run'),
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


def test_lazily_stage():
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

    expected = [Msg('subscribe', None, 'all', cb),
                Msg('null', None, 'test_arg', test_kwarg='val'),
                Msg('unsubscribe', token=None)]

    assert processed_plan == expected

    processed_plan = list(subs_decorator({'all': cb})(plan)('test_arg',
                                                            test_kwarg='val'))
    assert processed_plan == expected


def test_md():
    def plan():
        yield from open_run(md={'a': 1})

    processed_plan = list(inject_md_wrapper(plan(), {'b': 2}))

    expected = [Msg('open_run', None, a=1, b=2)]

    assert processed_plan == expected


def test_finalize():
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
    processed_plan = list(finalize_decorator(lambda: [Msg('read', det)])(plan)())
    expected = [Msg('null'), Msg('read', det)]
    assert processed_plan == expected

    # decorator does NOT accept list
    with pytest.raises(TypeError):
        list(finalize_decorator([Msg('read', det)])(plan)())

    # nor generator instance
    with pytest.raises(TypeError):
        list(finalize_decorator(cleanup_plan())(plan)())


def test_finalize_runs_after_error(fresh_RE):
    def plan():
        yield from [Msg('null')]
        raise Exception

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator
    try:
        fresh_RE(finalize_wrapper(plan(), [Msg('read', det)]))
    except Exception:
        pass # swallow the Exception; we are interested in msgs below

    expected = [Msg('null'), Msg('read', det)]

    assert msgs == expected


def test_reset_positions(fresh_RE):
    motor = Mover('a', {'a': lambda x: x}, {'x': 0})
    motor.set(5)

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    fresh_RE(reset_positions_wrapper(plan()))

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

    fresh_RE(reset_positions_wrapper(plan()))

    expected = [Msg('read', motor),
                Msg('set', motor, 8), Msg('set', motor, 5), Msg('wait')]

    for msg in msgs:
        msg.kwargs.pop('group', None)

    assert msgs == expected


def test_relative_set(fresh_RE):
    motor = Mover('a', {'a': lambda x: x}, {'x': 0})
    motor.set(5)

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    def plan():
        yield from (m for m in [Msg('set', motor, 8)])

    fresh_RE(relative_set_wrapper(plan()))

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

    fresh_RE(relative_set_wrapper(plan()))

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
            return NullStatus()

    det = DummyMover('det')
    det.count_time = DummySignal()

    def plan():
        yield from (m for m in [Msg('read', det)])

    msgs = []

    def accumulator(msg):
        msgs.append(msg)

    fresh_RE.msg_hook = accumulator

    fresh_RE(configure_count_time_wrapper(plan(), 7))

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


def test_count_delay_argument():
    # num=7 but delay only provides 5 entries
    with pytest.raises(ValueError):
        # count raises ValueError when delay generator is expired
        list(count([det], num=7, delay=(2**i for i in range(5))))

    # num=6 with 5 delays between should product 6 readings
    msgs = count([det], num=6, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 6

    # num=5 with 5 delays should produce 5 readings
    msgs = count([det], num=5, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 5

    # num=4 with 5 delays should produce 4 readings
    msgs = count([det], num=4, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 4

    # num=None with 5 delays should produce 6 readings
    msgs = count([det], num=None, delay=(2**i for i in range(5)))
    read_count = len([msg for msg in msgs if msg.command == 'read'])
    assert read_count == 6


def test_plan_md(fresh_RE):
    mutable = []
    md = {'color': 'red'}

    def collector(name, doc):
        mutable.append(doc)

    # test genereator
    mutable.clear()
    fresh_RE(count([det], md=md), collector)
    assert 'color' in mutable[0]

    # test Plan with explicit __init__
    mutable.clear()
    fresh_RE(Count([det], md=md), collector)
    assert 'color' in mutable[0]

    # test Plan with implicit __init__ (created via metaclasss)
    mutable.clear()
    fresh_RE(Scan([det], motor, 1, 2, 2, md=md), collector)
    assert 'color' in mutable[0]


def test_infinite_count(fresh_RE):
    loop = fresh_RE.loop

    loop.call_later(2, fresh_RE.stop)
    docs = defaultdict(list)

    def collector(name, doc):
        docs[name].append(doc)

    fresh_RE(count([det], num=None), collector)

    assert len(docs['start']) == 1
    assert len(docs['stop']) == 1
    assert len(docs['descriptor']) == 1
    assert len(docs['event']) > 0


def test_no_rewind_device():
    class FakeSig:
        def get(self):
            return False

    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    det.rewindable = FakeSig()

    assert not all_safe_rewind([det])


def test_monitor(fresh_RE):
    RE = fresh_RE
    RE(monitor_during_wrapper(count([det], 5), [det1]))
