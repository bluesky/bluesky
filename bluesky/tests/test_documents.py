import pytest
import jsonschema
from bluesky.run_engine import RunEngine
from event_model import DocumentNames, schemas
from bluesky.utils import new_uid
from bluesky.examples import simple_scan


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
    jsonschema.validate(doc, schemas[DocumentNames.start])
    # Add a legal key.
    doc.update({'b': 'c'})
    jsonschema.validate(doc, schemas[DocumentNames.start])
    # Now add illegal key.
    doc.update({'b.': 'c'})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schemas[DocumentNames.start])

    doc = {'time': 0,
           'uid': new_uid(),
           'data_keys': {'a': {'source': '',
                               'dtype': 'number',
                               'shape': []}},
           'run_start': new_uid()}
    jsonschema.validate(doc, schemas[DocumentNames.descriptor])
    # Add a legal key.
    doc.update({'b': 'c'})
    jsonschema.validate(doc, schemas[DocumentNames.descriptor])
    # Now add illegal key.
    doc.update({'b.c': 'd'})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schemas[DocumentNames.descriptor])

    doc = {'time': 0,
           'uid': new_uid(),
           'exit_status': 'success',
           'reason': '',
           'run_start': new_uid()}
    jsonschema.validate(doc, schemas[DocumentNames.stop])
    # Add a legal key.
    doc.update({'b': 'c'})
    jsonschema.validate(doc, schemas[DocumentNames.stop])
    # Now add illegal key.
    doc.update({'.b': 'c'})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schemas[DocumentNames.stop])
