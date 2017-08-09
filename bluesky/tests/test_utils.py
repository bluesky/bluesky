from bluesky.utils import ensure_generator, Msg
from bluesky.examples import stepscan, det, motor
import numpy as np

def test_single_msg_to_gen():
    m = Msg('set', None, 0)

    m_list = [m for m in ensure_generator(m)]

    assert len(m_list) == 1
    assert m_list[0] == m

def test_illegal_np(fresh_RE, db):
    RE = fresh_RE
    illegal_field = np.asarray(range(10))
    RE.subscribe(db.mds.insert)
    uid, = RE(stepscan(det, motor), group='foo', beamline_id='testing',
              config={}, illegal_field=illegal_field)
