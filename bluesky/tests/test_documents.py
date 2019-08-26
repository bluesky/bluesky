import pytest
import jsonschema
from event_model import DocumentNames, schema_validators
from bluesky.utils import new_uid
from bluesky.run_engine import Msg


def simple_scan(motor):
    yield Msg('open_run')
    yield Msg('set', motor, 5)
    yield Msg('read', motor)
    yield Msg('close_run')


def test_custom_metadata(RE, hw):
    def assert_lion(name, doc):
        assert 'animal' in doc
        assert doc['animal'] == 'lion'

    RE(simple_scan(hw.motor), {'start': assert_lion}, animal='lion')
    # Note: Because assert_lion is processed on the main thread, it can
    # fail the test. I checked by writing a failing version of it.  - D.A.


def test_dots_not_allowed_in_keys():
    doc = {'time': 0,
           'uid': new_uid()}
    schema_validators[DocumentNames.start].validate(doc)
    # Add a legal key.
    doc.update({'b': 'c'})
    schema_validators[DocumentNames.start].validate(doc)
    # Now add illegal key.
    doc.update({'b.': 'c'})
    with pytest.raises(jsonschema.ValidationError):
        schema_validators[DocumentNames.start].validate(doc)

    doc = {'time': 0,
           'uid': new_uid(),
           'data_keys': {'a': {'source': '',
                               'dtype': 'number',
                               'shape': []}},
           'run_start': new_uid()}
    schema_validators[DocumentNames.descriptor].validate(doc)
    # Add a legal key.
    doc.update({'b': 'c'})
    schema_validators[DocumentNames.descriptor].validate(doc)
    # Now add illegal key.
    doc.update({'b.c': 'd'})
    with pytest.raises(jsonschema.ValidationError):
        schema_validators[DocumentNames.descriptor].validate(doc)

    doc = {'time': 0,
           'uid': new_uid(),
           'exit_status': 'success',
           'reason': '',
           'run_start': new_uid()}
    schema_validators[DocumentNames.stop].validate(doc)
    # Add a legal key.
    doc.update({'b': 'c'})
    schema_validators[DocumentNames.stop].validate(doc)
    # Now add illegal key.
    doc.update({'.b': 'c'})
    with pytest.raises(jsonschema.ValidationError):
        schema_validators[DocumentNames.stop].validate(doc)
