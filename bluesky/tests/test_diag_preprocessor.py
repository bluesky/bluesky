import pickle
from bluesky import DiagnosticPreprocessor
from bluesky.plans import count, pchain
from bluesky.examples import det, det2, det3, TrivialFlyer


flyer1 = TrivialFlyer()
flyer2 = TrivialFlyer()


def test_monitors():
    # no-op
    D = DiagnosticPreprocessor()
    original = list(count([det]))
    processed = list(D(count([det])))
    assert len(processed) == len(original)

    # one monitor
    D.monitors.append(det2)
    original = list(count([det]))
    processed = list(D(count([det])))
    assert len(processed) == 2 + len(original)

    # two monitors
    D.monitors.append(det3)
    processed = list(D(count([det])))
    assert len(processed) == 4 + len(original)

    # two monitors applied a plan with consecutive runs
    original = list(list(count([det])) + list(count([det])))
    processed = list(D(pchain(count([det]), count([det]))))
    assert len(processed) == 8 + len(original)


def test_flyers():
    # one flyer
    D = DiagnosticPreprocessor(flyers=[flyer1])
    original = list(count([det]))
    processed = list(D(count([det])))
    # should add kickoff, wait, complete, wait, collect
    assert len(processed) == 5 + len(original)

    # two flyers
    D.flyers.append(flyer2)
    processed = list(D(count([det])))
    # should add 2 * (kickoff, complete, collect) + 2 * (wait)
    assert len(processed) == 8 + len(original)

    # two flyers applied to a plan with two consecutive runs
    original = list(list(count([det])) + list(count([det])))
    processed = list(D(pchain(count([det]), count([det]))))
    assert len(processed) == 16 + len(original)


def test_baseline():
    # one baseline detector
    D = DiagnosticPreprocessor(baseline=[det2])
    original = list(count([det]))
    processed = list(D(count([det])))
    # should add 2X (trigger, wait, create, read, save)
    assert len(processed) == 10 + len(original)

    # two baseline detectors
    D.baseline.append(det3)
    processed = list(D(count([det])))
    # should add 2X (trigger, triger, wait, create, read, read, save)
    assert len(processed) == 14 + len(original)

    # two baseline detectors applied to a plan with two consecutive runs
    original = list(list(count([det])) + list(count([det])))
    processed = list(D(pchain(count([det]), count([det]))))
    assert len(processed) == 28 + len(original)


def test_mixture():
    D = DiagnosticPreprocessor(baseline=[det2], flyers=[flyer1],
                               monitors=[det3])

    original = list(count([det]))
    processed = list(D(count([det])))
    assert len(processed) == 2 + 5 + 10 + len(original)


def test_repr():
    expected = "DiagnosticPreprocessor(baseline=[], monitors=[], flyers=[])"
    D = DiagnosticPreprocessor()
    actual = repr(D)
    assert actual == expected


def test_pickle():
    D = DiagnosticPreprocessor(baseline=['placeholder1'],
                               monitors=['placeholder2'],
                               flyers=['placeholder3'])
    D2 = pickle.loads(pickle.dumps(D))
    assert D2.baseline == D.baseline
    assert D2.monitors == D.monitors
    assert D2.flyers == D.flyers
