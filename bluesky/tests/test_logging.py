import logging

from bluesky.log import validate_level


def make_record(level, doc_name):
    levelno = validate_level(level)
    record = logging.getLogRecordFactory()('', levelno, '', '', '', '', '')
    if doc_name is not None:
        record.doc_name = doc_name
    return record


def test_noninterfering_handlers():

    class TestHandler(logging.Handler):
        "Collects all the records it emits in a list for checking in test."
        def __init__(self):
            self.records = []
            super().__init__()

        def emit(self, record):
            self.records.append(record)

    info_handler = TestHandler()
    info_handler.setLevel('INFO')
    debug_handler = TestHandler()
    debug_handler.setLevel('DEBUG')

    log = logging.getLogger('bluesky')
    log.setLevel('DEBUG')
    assert log.getEffectiveLevel() == 10
    sub_log = logging.getLogger('bluesky.test')
    log.addHandler(info_handler)
    log.addHandler(debug_handler)
    log.debug('test debug')
    log.info('test debug')
    assert len(info_handler.records) == 1
    assert len(debug_handler.records) == 2
    assert set(info_handler.records).issubset(debug_handler.records)

    info_handler.records.clear()
    debug_handler.records.clear()

    sub_log = logging.getLogger('bluesky.test')
    assert sub_log.getEffectiveLevel() == 10
    sub_log.debug('test debug')
    sub_log.info('test debug')
    assert len(info_handler.records) == 1
    assert len(debug_handler.records) == 2
    assert set(info_handler.records).issubset(debug_handler.records)
