from bluesky.utils import ensure_generator, Msg


def test_single_msg_to_gen():
    m = Msg('set', None, 0)

    m_list = [m for m in ensure_generator(m)]

    assert len(m_list) == 1
    assert m_list[0] == m
