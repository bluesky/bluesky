from distutils.version import LooseVersion
import pytest
import inspect
from bluesky.tests.utils import DocCollector
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.utils import RequestStop
import numpy as np
import numpy.testing as npt
import pandas as pd
import random
import re
import collections
from bluesky.tests.utils import MsgCollector
from bluesky.plan_patterns import chunk_outer_product_args, outer_product


def _validate_start(start, expected_values):
    '''Basic metadata validtion'''

    plan_md_key = [
        'plan_pattern_module',
        'plan_pattern_args',
        'plan_type',
        'plan_pattern',
        'plan_name',
        'num_points',
        'plan_args',
        'detectors']

    for k in plan_md_key:
        assert k in start
    for k, v in expected_values.items():
        assert start[k] == v


def test_plan_header(RE, hw):
    args = []

    ##
    args.append((bp.grid_scan([hw.det],
                              hw.motor, 1, 2, 3,
                              hw.motor1, 4, 5, 6,
                              hw.motor2, 7, 8, 9,
                              snake_axes=True),
                 {'motors': ('motor', 'motor1', 'motor2'),
                  'extents': ([1, 2], [4, 5], [7, 8]),
                  'shape': (3, 6, 9),
                  'snaking': (False, True, True),
                  'plan_pattern_module': 'bluesky.plan_patterns',
                  'plan_pattern': 'outer_product',
                  'plan_name': 'grid_scan'}))

    ##
    args.append((bp.inner_product_scan([hw.det], 9,
                                       hw.motor, 1, 2,
                                       hw.motor1, 4, 5,
                                       hw.motor2, 7, 8),
                {'motors': ('motor', 'motor1', 'motor2')}))

    for plan, target in args:
        c = DocCollector()
        RE(plan, c.insert)
        for s in c.start:
            _validate_start(s, target)


def test_ops_dimension_hints(RE, hw):
    det = hw.det
    motor = hw.motor
    motor1 = hw.motor1
    c = DocCollector()
    RE.subscribe(c.insert)
    RE(bp.grid_scan([det],
                    motor, -1, 1, 7,
                    motor1, 0, 2, 3))

    st = c.start[0]

    assert 'dimensions' in st['hints']

    assert st['hints']['dimensions'] == [
        (m.hints['fields'], 'primary') for m in (motor, motor1)]


def test_mesh_pseudo(hw, RE):

    p3x3 = hw.pseudo3x3
    sig = hw.sig
    d = DocCollector()

    RE.subscribe(d.insert)
    rs = RE(bp.grid_scan([sig],
                         p3x3.pseudo1, 0, 3, 5,
                         p3x3.pseudo2, 7, 10, 7))

    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]

    df = pd.DataFrame([_['data']
                       for _ in d.event[d.descriptor[uid][0]['uid']]])

    for k in p3x3.describe():
        assert k in df

    for k in sig.describe():
        assert k in df

    assert all(df[sig.name] == 0)
    assert all(df[p3x3.pseudo3.name] == 0)


def test_rmesh_pseudo(hw, RE):
    p3x3 = hw.pseudo3x3
    p3x3.set(1, -2, 100)
    init_pos = p3x3.position
    sig = hw.sig
    d = DocCollector()

    RE.subscribe(d.insert)
    rs = RE(bp.rel_grid_scan([sig],
                             p3x3.pseudo1, 0, 3, 5,
                             p3x3.pseudo2, 7, 10, 7))

    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]

    df = pd.DataFrame([_['data']
                       for _ in d.event[d.descriptor[uid][0]['uid']]])

    for k in p3x3.describe():
        assert k in df

    for k in sig.describe():
        assert k in df

    assert all(df[sig.name] == 0)
    assert all(df[p3x3.pseudo3.name] == 100)
    assert len(df) == 35
    assert min(df[p3x3.pseudo1.name]) == 1
    assert init_pos == p3x3.position


def test_relative_pseudo(hw, RE, db):

    RE.subscribe(db.insert)
    p = hw.pseudo3x3
    p.set(1, 1, 1)
    base_pos = p.position

    # this triggers the merging code path
    rs = RE(bp.relative_inner_product_scan([p],
                                           5,
                                           p.pseudo1, -1, 1,
                                           p.pseudo2, -2, -1))

    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]

    tb1 = db[uid].table().drop('time', axis=1)
    assert p.position == base_pos

    # this triggers this does not
    rs = RE(bp.relative_inner_product_scan([p],
                                           5,
                                           p.real1, 1, -1,
                                           p.real2, 2, 1))
    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]

    tb2 = db[uid].table().drop('time', axis=1)
    assert p.position == base_pos

    # same columns
    assert set(tb1) == set(tb2)
    # same number of points
    assert len(tb1) == len(tb2)

    def get_hint(c):
        h = c.hints['fields']
        return h[0] if h else c.name

    for c in list(p.pseudo_positioners) + list(p.real_positioners):
        col = get_hint(c)
        print(col)
        assert (tb1[col] == tb2[col]).all()

    assert (tb1[get_hint(p.pseudo1)] == np.linspace(0, 2, 5)).all()


def test_reset_wrapper(hw, RE, monkeypatch):
    monkeypatch.setenv('BLUESKY_PREDECLARE', '1')
    p = hw.pseudo3x3
    m_col = MsgCollector()
    RE.msg_hook = m_col

    RE(bp.relative_inner_product_scan([], 1,
                                      p.pseudo1, 0, 1,
                                      p.pseudo2, 0, 1))
    expecte_objs = [p, None, None, None,
                    p, None, p,
                    None, None, p,
                    None, None, p,
                    p, None]
    assert len(m_col.msgs) == 15
    assert [m.obj for m in m_col.msgs] == expecte_objs


@pytest.mark.parametrize('pln', [bps.mv, bps.mvr])
def test_pseudo_mv(hw, RE, pln):
    p = hw.pseudo3x3
    m_col = MsgCollector()
    RE.msg_hook = m_col

    RE(pln(p.pseudo1, 1,
           p.pseudo2, 1))
    expecte_objs = [p, None]
    assert len(m_col.msgs) == 2
    assert [m.obj for m in m_col.msgs] == expecte_objs


def _good_per_step_factory():
    def per_step_old(detectors, motor, step):
        yield from bps.null()

    def per_step_extra(detectors, motor, step, some_kwarg=None):
        yield from bps.null()

    def per_step_exact(detectors, motor, step, take_reading=None):
        yield from bps.null()

    def per_step_kwargs(detectors, motor, step, **kwargs):
        yield from bps.null()

    def per_nd_step(detectors, post_cache, *args, **kwargs):
        yield from bps.null()

    return pytest.mark.parametrize(
        "per_step",
        [per_step_old, per_step_extra, per_step_exact, per_step_kwargs],
        ids=["no kwargs", "extra kwargs", "exact signature", "with kwargs"],
    )


@_good_per_step_factory()
def test_good_per_step_signature(hw, per_step):

    list(bp.scan([hw.det], hw.motor, -1, 1, 5, per_step=per_step))


def _bad_per_step_factory():
    def too_few(detectors, motor):
        "no body"

    def too_many(detectors, motor, step, bob):
        "no body"

    def extra_required_kwarg(detectors, motor, step, *, some_kwarg):
        "no body"

    def wrong_names(a, b, c, take_reading=None):
        "no body"

    def per_step_only_args(*args):
        "no body"

    def per_nd_step_extra(detectors, step, pos_cache, extra_no_dflt):
        "no body"

    def per_nd_step_bad_pos(detectors, step, pos_cache, *, extra_no_dflt):
        "no body"

    def all_wrong(a, b, c=None, *args, d=None, g, **kwargs):
        "no body"

    return pytest.mark.parametrize(
        "per_step",
        [too_few, too_many, extra_required_kwarg, wrong_names, per_step_only_args],
        ids=["too few", "too many", "required kwarg", "bad name", "args only"],
    )


@_bad_per_step_factory()
def test_bad_per_step_signature(hw, per_step):
    sig = inspect.signature(per_step)
    print(f'*** test bad_per_step {sig} ***\n')
    with pytest.raises(
        TypeError,
        match=re.escape(
           "per_step must be a callable with the signature \n "
            "<Signature (detectors, step, pos_cache)> or "
            "<Signature (detectors, motor, step)>. \n"
            "per_step signature received: {}".format(sig)
        ),
    ):
        list(bp.scan([hw.det], hw.motor, -1, 1, 5, per_step=per_step))


def require_ophyd_1_4_0():
    ophyd = pytest.importorskip("ophyd")
    if LooseVersion(ophyd.__version__) < LooseVersion('1.4.0'):
        pytest.skip("Needs ophyd 1.4.0 for realistic ophyd.sim Devices.")


@pytest.mark.parametrize("val", [0, None, "aardvark"])
def test_rd_dflt(val):
    ophyd = pytest.importorskip("ophyd")
    require_ophyd_1_4_0()
    sig = ophyd.Signal(value="0", name="sig")

    def tester(obj, dflt):
        ret = yield from bps.rd(obj, default_value=dflt)
        assert ret is dflt

    list(tester(sig, val))


@pytest.mark.parametrize("val", [0, None, "aardvark"])
def test_rd(RE, val):
    ophyd = pytest.importorskip("ophyd")
    require_ophyd_1_4_0()
    sig = ophyd.Signal(value="0", name="sig")

    def tester(obj, val):
        yield from bps.mv(sig, val)
        ret = yield from bps.rd(obj, default_value=object())
        assert ret == val

    RE(tester(sig, val))


def test_rd_fails(hw):
    require_ophyd_1_4_0()
    obj = hw.det

    obj.noise.kind = "hinted"
    hints = obj.hints.get("fields", [])
    msg = re.escape(
        f"Your object {obj} ({obj.name}.{obj.dotted_name}) "
        + f"has {len(hints)} items hinted ({hints}).  We "
    )
    with pytest.raises(ValueError, match=msg):
        list(bps.rd(obj))

    obj.noise.kind = "normal"
    obj.val.kind = "normal"
    msg = re.escape(
        f"Your object {obj} ({obj.name}.{obj.dotted_name}) "
        + f"and has {len(obj.read_attrs)} read attrs.  We "
    )
    with pytest.raises(ValueError, match=msg):
        list(bps.rd(obj))

    obj.read_attrs = []

    msg = re.escape(
        f"Your object {obj} ({obj.name}.{obj.dotted_name}) "
        + f"and has {len(obj.read_attrs)} read attrs.  We "
    )
    with pytest.raises(ValueError, match=msg):
        list(bps.rd(obj))


@pytest.mark.parametrize("kind", ["hinted", "normal"])
def test_rd_device(hw, RE, kind):
    require_ophyd_1_4_0()
    called = False
    hw.det.val.kind = kind

    def tester(obj):
        nonlocal called
        direct_read = yield from bps.read(obj)
        rd_read = yield from bps.rd(obj)
        sig_read = yield from bps.rd(obj.val)

        assert rd_read == direct_read["det"]["value"]
        assert sig_read == rd_read
        called = True

    RE(tester(hw.det))
    assert called


# ********  Tests for `grid_scan` and `rel_grid_scan` plans  ***********


def _retrieve_motor_positions(doc_collector, motors):
    """
    Retrieves the motor positions for the completed run.

    Parameters
    ----------
    `doc_collector`: DocCollector
        DocCollector object that contains data from a single run
    `motors`: list
        the list of motors for which positions should be collected.

    Returns
    -------
    the dictionary:
    {'motor_name_1': list of positions,
     'motor_name_2': list of positions, ...}

    """
    motor_names = [_.name for _ in motors]
    # Get the event list for the first run
    desc = next(iter(doc_collector.event.keys()))  # Descriptor
    event_list = doc_collector.event[desc]

    # Now collect the positions
    positions = {k: [] for k in motor_names}
    for event in event_list:
        for name in positions.keys():
            positions[name].append(event["data"][name])

    return positions


def _grid_scan_position_list(args, snake_axes):
    """
    Generates the lists of positions for each motor during the 'grid_scan'.

    Parameters
    ----------
    args: list
        list of arguments, same as the parameter `args` of the `grid_scan`
    snake_axes: None, True, False or iterable
        same meaning as `snake_axes` parameter of the `grid_scan`

    Returns
    -------
    Tuple of the dictionary:
        {'motor_name_1': list of positions,
         'motor_name_2': list of positions, ...}
    and the tuple that lists `snaking` status of each motor (matches the contents
    of the 'snaking' field of the start document.
    """
    # If 'snake_axis' is specified, it always overwrides any snaking values specified in 'args'
    chunk_args = list(chunk_outer_product_args(args))
    if isinstance(snake_axes, collections.abc.Iterable):
        for n, chunk in enumerate(chunk_args):
            motor, start, stop, num, snake = chunk
            if motor in snake_axes:
                chunk_args[n] = tuple([motor, start, stop, num, True])
            else:
                chunk_args[n] = tuple([motor, start, stop, num, False])
    elif snake_axes is True:
        chunk_args = [(motor, start, stop, num, True) if n > 0
                      else (motor, start, stop, num, False)
                      for n, (motor, start, stop, num, _) in enumerate(chunk_args)]
    elif snake_axes is False:
        chunk_args = [(motor, start, stop, num, False)
                      for (motor, start, stop, num, _) in chunk_args]
    elif snake_axes is None:
        pass
    else:
        raise ValueError(f"The value of 'snake_axes' is not iterable, boolean or None: '{snake_axes}'")

    # Expected contents of the 'snaking' field in the start document
    snaking = tuple([_[4] for _ in chunk_args])

    # Now convert the chunked argument list to regular argument list before calling the cycler
    args_modified = []
    for n, chunk in enumerate(chunk_args):
        if n > 0:
            args_modified.extend(chunk)
        else:
            args_modified.extend(chunk[:-1])

    # Note, that outer_product is used to generate the list of coordinate points
    #   while the plan is executed, but it is tested elsewhere, so it can be trusted
    full_cycler = outer_product(args=args_modified)
    event_list = list(full_cycler)

    # The list of motors
    motors = [_[0] for _ in chunk_args]
    motor_names = [_.name for _ in motors]

    positions = {k: [] for k in motor_names}
    for event in event_list:
        for m, mn in zip(motors, motor_names):
            positions[mn].append(event[m])

    return positions, snaking


@pytest.mark.parametrize("args, snake_axes", [
    # Calls using new arguments
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6,
      "motor2", 7, 8, 9),
     None),
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6,
      "motor2", 7, 8, 9),
     False),
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6,
      "motor2", 7, 8, 9),
     True),
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6,
      "motor2", 7, 8, 9),
     []),  # Empty list will disable snaking
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6,
      "motor2", 7, 8, 9),
     ["motor1"]),
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6,
      "motor2", 7, 8, 9),
     ["motor2"]),
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6,
      "motor2", 7, 8, 9),
     ["motor1", "motor2"]),

    # Deprecated calls
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6, True,
      "motor2", 7, 8, 9, True),
     None),  # snake_axes may be only set to None
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6, True,
      "motor2", 7, 8, 9, False),
     None),
    (("motor", 1, 2, 3,
      "motor1", 4, 5, 6, False,
      "motor2", 7, 8, 9, True),
     None),
])
@pytest.mark.parametrize("plan, is_relative", [
    (bp.grid_scan, False),
    (bp.rel_grid_scan, True)
])
def test_grid_scans(RE, hw, args, snake_axes, plan, is_relative):
    """
    Basic test of functionality of `grid_scan` and `rel_grid_scan`:
    Tested:
    - positions of the simulated motors at each step of the scan
    - contents of the 'snaking' field of the start document
    """

    # Convert motor names to actual motors in the argument list using fixture 'hw'
    args = [getattr(hw, _) if isinstance(_, str) else _ for _ in args]
    # Do the same in `snake_axes` if it contains the list of motors
    if isinstance(snake_axes, collections.abc.Iterable):
        snake_axes = [getattr(hw, _) for _ in snake_axes]

    # Place motors at random initial positions. Do it both for relative and
    #   absolute scans. The absolute scans will ignore the inital positions
    #   automatically.
    motors = [_[0] for _ in chunk_outer_product_args(args)]
    motors_pos = [2 * random.random() - 1 for _ in range(len(motors))]
    for _motor, _pos in zip(motors, motors_pos):
        RE(bps.mv(_motor, _pos))

    c = DocCollector()
    RE(plan([hw.det], *args, snake_axes=snake_axes), c.insert)
    positions = _retrieve_motor_positions(c, [hw.motor, hw.motor1, hw.motor2])
    # Retrieve snaking data from the start document
    snaking = c.start[0]["snaking"]

    # Generate the list of positions based on
    positions_expected, snaking_expected = \
        _grid_scan_position_list(args=args, snake_axes=snake_axes)

    assert snaking == snaking_expected, \
        "The contents of the 'snaking' field in the start document "\
        "does not match the expected values"

    assert set(positions.keys()) == set(positions_expected.keys()), \
        "Different set of motors in dictionaries of actual and expected positions"

    # The dictionary of the initial postiions
    motor_pos_shift = {_motor.name: _pos for (_motor, _pos) in zip(motors, motors_pos)}

    for name in positions_expected.keys():
        # The positions should be shifted only if the plan is relative.
        #   Absolute plans will ignore the initial motor positions
        shift = motor_pos_shift[name] if is_relative else 0
        npt.assert_array_almost_equal(
            positions[name], np.array(positions_expected[name]) + shift,
            err_msg=f"Expected and actual positions for the motor '{name}' don't match")


@pytest.mark.parametrize("plan", [
    bp.grid_scan,
    bp.rel_grid_scan
])
def test_grid_scans_failing(RE, hw, plan):
    """Test the failing cases of 'grid_scan' and 'rel_grid_scan'"""

    # Multiple instance of the same motor in 'args'
    args_list = [
        # New style
        (hw.motor, 1, 2, 3,
         hw.motor1, 4, 5, 6,
         hw.motor1, 7, 8, 9),
        # Old style
        (hw.motor, 1, 2, 3,
         hw.motor1, 4, 5, 6, True,
         hw.motor1, 7, 8, 9, False)
    ]
    for args in args_list:
        with pytest.raises(ValueError,
                           match="Some motors are listed multiple times in the argument list 'args'"):
            RE(plan([hw.det], *args))

    # 'snake_axes' contains repeated elements
    with pytest.raises(ValueError,
                       match="The list of axes 'snake_axes' contains repeated elements"):
        args = (hw.motor, 1, 2, 3,
                hw.motor1, 4, 5, 6,
                hw.motor2, 7, 8, 9)
        snake_axes = [hw.motor1, hw.motor2, hw.motor1]
        RE(plan([hw.det], *args, snake_axes=snake_axes))

    # Snaking is enabled for the slowest motor
    with pytest.raises(ValueError,
                       match="The list of axes 'snake_axes' contains the slowest motor"):
        args = (hw.motor, 1, 2, 3,
                hw.motor1, 4, 5, 6,
                hw.motor2, 7, 8, 9)
        snake_axes = [hw.motor1, hw.motor]
        RE(plan([hw.det], *args, snake_axes=snake_axes))

    # Attempt to enable snaking for motors that are not controlled during the scan
    with pytest.raises(ValueError,
                       match="The list of axes 'snake_axes' contains motors "
                             "that are not controlled during the scan"):
        args = (hw.motor, 1, 2, 3,
                hw.motor1, 4, 5, 6,
                hw.motor2, 7, 8, 9)
        snake_axes = [hw.motor1, hw.motor3]
        RE(plan([hw.det], *args, snake_axes=snake_axes))

    # Mix deprecated and new API ('snake_axes' is used while snaking is set in 'args'
    with pytest.raises(ValueError,
                       match="Mixing of deprecated and new API interface is not allowed"):
        args = (hw.motor, 1, 2, 3,
                hw.motor1, 4, 5, 6, True,
                hw.motor2, 7, 8, 9, False)
        RE(plan([hw.det], *args, snake_axes=False))

    # The type of 'snake_axes' parameter is not allowed
    for snake_axes in (10, 50.439, "some string"):
        with pytest.raises(ValueError,
                           match="Parameter 'snake_axes' is not iterable, boolean or None"):
            args = (hw.motor, 1, 2, 3,
                    hw.motor1, 4, 5, 6,
                    hw.motor2, 7, 8, 9)
            RE(plan([hw.det], *args, snake_axes=snake_axes))


def test_describe_failure(RE):
    ophyd = pytest.importorskip("ophyd")

    class Aardvark(Exception):
        # Unique exception to test behavior of describe.
        ...

    class BadSignalDescribe(ophyd.Signal):
        def describe(self):
            raise Aardvark("Look, an aardvark!")

    class BadSignalRead(ophyd.Signal):
        def read(self):
            raise Aardvark("Look, the other aardvark!")

    bad_signal1 = BadSignalDescribe(value=5, name="Arty")
    bad_signal2 = BadSignalRead(value=5, name="Arty")
    good_signal = ophyd.Signal(value=42, name='baseline')

    class StreamTester:
        def __init__(self):
            self.event_count = 0
            self.stream_names = set()

        def __call__(self, name, doc):
            if name == 'event':
                self.event_count += 1
            if name == 'descriptor':
                self.stream_names.add(doc['name'])

        def verify(self):
            assert self.event_count == 2
            assert self.stream_names == set(['baseline'])
    st = StreamTester()
    with pytest.raises(Aardvark, match="Look, an aardvark!"):
        RE(
            bpp.baseline_wrapper(
                bp.count([bad_signal1]),
                [good_signal]
            ),
            st
        )
    st.verify()

    st = StreamTester()
    with pytest.raises(Aardvark, match="Look, the other aardvark!"):
        RE(
            bpp.baseline_wrapper(
                bp.count([bad_signal2]),
                [good_signal]
            ),
            st
        )
    st.verify()


def test_errors_through_msg_mutator(hw, monkeypatch):
    monkeypatch.setenv('BLUESKY_PREDECLARE', '1')
    gen = bp.rel_scan([], hw.motor, 5, -5, 10)

    msgs = []

    msgs.append(next(gen))
    msgs.append(next(gen))
    msgs.append(next(gen))
    msgs.append(next(gen))
    msgs.append(next(gen))
    msgs.append(gen.throw(RequestStop))
    try:
        while True:
            msgs.append(next(gen))
    except RequestStop:
        pass
    else:
        raise False

    target = [
        "stage",
        "open_run",
        "declare_stream",
        "checkpoint",
        "set",
        "close_run",
        "unstage",
        "set",
        "wait",
    ]

    assert target == [m.command for m in msgs]


@pytest.mark.parametrize('predeclare', [True, False])
def test_predeclare_env(hw, monkeypatch, predeclare):
    from cycler import cycler

    if predeclare:
        monkeypatch.setenv('BLUESKY_PREDECLARE', '1')

    for p in [bp.count([hw.det]),
              bp.scan_nd([hw.det], cycler(hw.motor1, [1, 2, 3]) * cycler(hw.motor2, [4, 5, 6])),
              bp.log_scan([hw.det], hw.motor, 1, 20, 5)]:
        cmds = [m.command for m in p]
        if predeclare:
            assert 'declare_stream' in cmds
        else:
            assert 'declare_stream' not in cmds
