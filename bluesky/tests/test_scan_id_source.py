import bluesky.plans as bp


def test_scan_id_source_two_REs(hw, RE, RE_no_scan_id, db):
    RE.subscribe(db.insert)
    RE_no_scan_id.subscribe(db.insert)
    det = hw.det

    uid1, = RE(bp.count([det], num=3))
    uid2, = RE_no_scan_id(bp.count([det], num=3))
    uid3, = RE(bp.count([det], num=3))

    hdr1 = db[uid1]
    hdr2 = db[uid2]
    hdr3 = db[uid3]

    # First scan, we have 'scan_id' in the start doc and it's 1.
    assert 'scan_id' in hdr1['start']
    assert hdr1['start']['scan_id'] == 1

    # Second scan, we don't have 'scan_id' in the start doc.
    assert 'scan_id' not in hdr2['start']

    # Third scan, we have 'scan_id' in the start doc and it's 2.
    assert 'scan_id' in hdr3['start']
    assert hdr3['start']['scan_id'] == 2


def test_scan_id_source_one_RE(hw, RE, db):
    RE.subscribe(db.insert)
    det = hw.det

    uid1, = RE(bp.count([det], num=3))

    RE.scan_id_source = None
    uid2, = RE(bp.count([det], num=3))

    from bluesky.run_engine import default_scan_id_source
    RE.scan_id_source = default_scan_id_source
    uid3, = RE(bp.count([det], num=3))

    hdr1 = db[uid1]
    hdr2 = db[uid2]
    hdr3 = db[uid3]

    # First scan, we have 'scan_id' in the start doc and it's 1.
    assert 'scan_id' in hdr1['start']
    assert hdr1['start']['scan_id'] == 1

    # Second scan, we don't have 'scan_id' in the start doc.
    assert 'scan_id' not in hdr2['start']

    # Third scan, we have 'scan_id' in the start doc and it's 2.
    assert 'scan_id' in hdr3['start']
    assert hdr3['start']['scan_id'] == 2

