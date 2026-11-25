import pytest
from bluesky.pydantic_md import Md
from pydantic import ValidationError
from collections import ChainMap

class NonSerializableExample:
    def __init__(self, var):
        self.var = var

@pytest.fixture
def non_serializable_instance():
    return NonSerializableExample(3)

@pytest.fixture
def md_instance():
    return Md(key1='initial', key2=42)

def test_initialization(md_instance):
    assert md_instance['key1'] == 'initial'
    assert md_instance['key2'] == 42
    assert md_instance['key3'] == 0.0

def test_setitem(md_instance):
    md_instance['key4'] = 'new value'
    assert md_instance['key4'] == 'new value'

def test_delitem(md_instance):
    md_instance['key4'] = 'new value'
    del md_instance['key4']
    with pytest.raises(AttributeError):
        _ = md_instance['key4']

def test_len(md_instance):
    assert len(md_instance) == 3
    md_instance['key4'] = 'new value'
    assert len(md_instance) == 4

def test_serialization(md_instance, non_serializable_instance):
    with pytest.raises(ValidationError):
        md_instance['key2'] = non_serializable_instance

def test_validation(md_instance):
    with pytest.raises(ValidationError):
        md_instance['key2'] = 'string'

def test_model_dump_json(md_instance):
    md_instance['key4'] = {1, 2, 3}
    json_str = md_instance.model_dump_json()
    assert '"key4":[1,2,3]' in json_str

def test_chainmap(md_instance):
    additional_md = Md(key4='additional')
    combined_md = ChainMap(md_instance.model_dump(), additional_md.model_dump())
    combined_md = Md(**combined_md)
    assert combined_md['key1'] == 'initial'
    assert combined_md['key4'] == 'additional'

def test_update(md_instance):
    bar = {'key1': 'new value', 'key4': 43}
    md_instance.update(bar)
    assert md_instance['key1'] == 'new value'
    assert md_instance['key4'] == 43