from collections import deque
import pytest
from bluesky import Msg
from bluesky.examples import det, det1, det2
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
                           repeater, repeater, caching_repeater)


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
