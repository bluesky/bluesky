from bluesky.examples import motor
from bluesky.plans import scan


def test_hints(fresh_RE):
    RE = fresh_RE
    expected_hint = {'axes': [motor.name]}
    assert motor.hints() == expected_hint
    collector = []

    def collect(*args):
        collector.append(args)

    RE(scan([], motor, 1, 2, 2), {'descriptor': collect})
    name, doc = collector.pop()
    assert doc['hints'][motor.name] == expected_hint
