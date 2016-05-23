from bluesky import Msg
from bluesky.callbacks.olog import logbook_cb_factory

text = []


def f(**kwargs):
    text.append(kwargs['text'])


def test_default_template(fresh_RE):
    text.clear()
    fresh_RE.subscribe('start', logbook_cb_factory(f))
    fresh_RE([Msg('open_run', plan_args={}), Msg('close_run')])
    assert len(text[0]) > 0


def test_trivial_template(fresh_RE):
    text.clear()
    fresh_RE.subscribe('start', logbook_cb_factory(f, desc_template='hello'))
    fresh_RE([Msg('open_run', plan_args={}), Msg('close_run')])
    assert text[0] == 'hello'

    # smoke test the long_template
    fresh_RE.subscribe('start', logbook_cb_factory(f, long_template='hello'))
    fresh_RE([Msg('open_run', plan_args={}), Msg('close_run')])

def test_template_dispatch(fresh_RE):
    disp = {'a': 'A', 'b': 'B'}
    text.clear()
    fresh_RE.subscribe('start', logbook_cb_factory(f, desc_dispatch=disp))
    fresh_RE([Msg('open_run', plan_name='a', plan_args={}),
              Msg('close_run')])
    fresh_RE([Msg('open_run', plan_name='b', plan_args={}),
              Msg('close_run')])
    assert text[0] == 'A'
    assert text[1] == 'B'

    # smoke test the long_dispatch
    fresh_RE.subscribe('start', logbook_cb_factory(f, long_dispatch=disp))
    fresh_RE([Msg('open_run', plan_name='a', plan_args={}),
              Msg('close_run')])
    fresh_RE([Msg('open_run', plan_name='b', plan_args={}),
              Msg('close_run')])
