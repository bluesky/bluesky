import logging
from bluesky.log import DocFilter, validate_level


def make_record(level, doc_name):
    levelno = validate_level(level)
    record = logging.getLogRecordFactory()('', levelno, '', '', '', '', '')
    if doc_name is not None:
        record.doc_name = doc_name
    return record


def test_over_barrier():
    "All records that meet or exceed the filter's level should pass."
    filter = DocFilter('descriptor', level='WARNING', exclusive=False)

    # Test relevant record that meets condition.
    record = make_record('WARNING', 'descriptor')
    assert filter.filter(record)
    record = make_record('ERROR', 'descriptor')
    assert filter.filter(record)

    # Test relevant record that fails condition.
    record = make_record('WARNING', 'event')
    assert filter.filter(record)
    record = make_record('ERROR', 'event')
    assert filter.filter(record)

    # Test not relevant record.
    record = make_record('WARNING', None)
    assert filter.filter(record)
    record = make_record('ERROR', None)
    assert filter.filter(record)

    # Test not relevant record on exclusive filter. Should be same result.
    ex_filter = DocFilter('descriptor', level='WARNING', exclusive=True)
    record = make_record('WARNING', None)
    assert ex_filter.filter(record)
    record = make_record('ERROR', None)
    assert ex_filter.filter(record)


def test_not_relevant():
    "When a filter is exclusive, not relevant records below barrier bounce."
    # A not relevant record does not pass through an exclusive filter.
    ex_filter = DocFilter('descriptor', level='WARNING', exclusive=True)
    record = make_record('INFO', None)
    assert not ex_filter.filter(record)

    # A not relevant record passes through a nonexclusive filter.
    filter = DocFilter('descriptor', level='WARNING', exclusive=False)
    record = make_record('INFO', None)
    assert filter.filter(record)


def test_relevant():
    filter = DocFilter('descriptor', level='WARNING', exclusive=False)
    record = make_record('INFO', 'event')
    assert not filter.filter(record)
    record = make_record('INFO', 'descriptor')
    assert filter.filter(record)

    # Same results for exclusive filter.
    ex_filter = DocFilter('descriptor', level='WARNING', exclusive=True)
    record = make_record('INFO', 'event')
    assert not ex_filter.filter(record)
    record = make_record('INFO', 'descriptor')
    assert ex_filter.filter(record)


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
