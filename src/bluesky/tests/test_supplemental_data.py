import pickle
from bluesky import SupplementalData
from bluesky.plans import count
from bluesky.preprocessors import pchain
from bluesky.utils import Msg


def test_monitors(hw):
    det = hw.det
    rand = hw.rand
    rand2 = hw.rand2

    # no-op
    D = SupplementalData()
    original = list(count([det]))
    processed = list(D(count([det])))
    assert len(processed) == len(original)

    # one monitor
    D.monitors.append(rand)
    original = list(count([det]))
    processed = list(D(count([det])))
    assert len(processed) == 2 + len(original)

    # two monitors
    D.monitors.append(rand2)
    processed = list(D(count([det])))
    assert len(processed) == 4 + len(original)

    # two monitors applied a plan with consecutive runs
    original = list(list(count([det])) + list(count([det])))
    processed = list(D(pchain(count([det]), count([det]))))
    assert len(processed) == 8 + len(original)


def test_flyers(hw):
    # one flyer
    D = SupplementalData(flyers=[hw.flyer1])
    original = list(count([hw.det]))
    processed = list(D(count([hw.det])))
    # should add kickoff, wait, complete, wait, collect
    assert len(processed) == 5 + len(original)

    # two flyers
    D.flyers.append(hw.flyer2)
    processed = list(D(count([hw.det])))
    # should add 2 * (kickoff, complete, collect) + 2 * (wait)
    assert len(processed) == 8 + len(original)

    # two flyers applied to a plan with two consecutive runs
    original = list(list(count([hw.det])) + list(count([hw.det])))
    processed = list(D(pchain(count([hw.det]), count([hw.det]))))
    assert len(processed) == 16 + len(original)


def test_baseline(hw):
    # one baseline detector
    D = SupplementalData(baseline=[hw.det2])
    original = list(count([hw.det]))
    processed = list(D(count([hw.det])))
    # should add 2X (trigger, wait, create, read, save)
    assert len(processed) == 11 + len(original)

    # two baseline detectors
    D.baseline.append(hw.det3)
    processed = list(D(count([hw.det])))
    # should add 2X (trigger, triger, wait, create, read, read, save)
    assert len(processed) == 15 + len(original)

    # two baseline detectors applied to a plan with two consecutive runs
    original = list(list(count([hw.det])) + list(count([hw.det])))
    processed = list(D(pchain(count([hw.det]), count([hw.det]))))
    assert len(processed) == 30 + len(original)


def test_mixture(hw):
    D = SupplementalData(baseline=[hw.det2],
                         flyers=[hw.flyer1],
                         monitors=[hw.rand])

    original = list(count([hw.det]))
    processed = list(D(count([hw.det])))
    assert len(processed) == 3 + 5 + 10 + len(original)


def test_order(hw):
    D = SupplementalData(baseline=[hw.det2],
                         flyers=[hw.flyer1],
                         monitors=[hw.rand])

    def null_run():
        yield Msg('open_run')
        yield Msg('null')
        yield Msg('close_run')

    actual = [msg.command for msg in D(null_run())]
    expected = ['open_run',
                # baseline
                'declare_stream',
                'trigger',
                'wait',
                'create',
                'read',
                'save',
                # monitors
                'monitor',
                # flyers
                'kickoff',
                'wait',
                # plan
                'null',
                # flyers
                'complete',
                'wait',
                'collect',
                # montiors
                'unmonitor',
                # baseline
                'trigger',
                'wait',
                'create',
                'read',
                'save',
                'close_run']
    assert actual == expected


def test_repr():
    expected = "SupplementalData(baseline=[], monitors=[], flyers=[])"
    D = SupplementalData()
    actual = repr(D)
    assert actual == expected


def test_pickle():
    D = SupplementalData(baseline=['placeholder1'],
                         monitors=['placeholder2'],
                         flyers=['placeholder3'])
    D2 = pickle.loads(pickle.dumps(D))
    assert D2.baseline == D.baseline
    assert D2.monitors == D.monitors
    assert D2.flyers == D.flyers


def test_uid_passthrough(RE, hw):
    # Test that none of the preprocessors in SupplementalData break plans
    # returning uids (a bug that was caught on the floor).

    RE.msg_hook = print

    # preliminary sanity check
    def mycount1():
        uid = yield from count([])
        assert uid is not None
        assert isinstance(uid, str)

    RE(mycount1())

    # actual test
    sd = SupplementalData()
    sd.baseline = [hw.det]
    sd.monitors = [hw.rand]
    sd.flyers = [hw.flyer1]
    hw.flyer1.loop = RE.loop

    def mycount2():
        uid = yield from sd(count([]))
        assert uid is not None
        assert isinstance(uid, str)

    RE(mycount2())
