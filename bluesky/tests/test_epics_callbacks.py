from nose.tools import assert_equal
import epics
p = None


def test_epics_smoke():

    print(p)
    pv = epics.PV('BSTEST:VAL')
    pv.value = 123456
    print(pv)
    print(pv.value)
    print(pv.connect())
    assert pv.connect()
    for j in range(1, 15):
        pv.put(j, wait=True)
        ret = pv.get(use_monitor=False)
        assert_equal(ret, j)
