from collections.abc import MutableMapping

import event_model
import ophyd

import bluesky
from bluesky.plans import count


def test_blueskyversion(RE):
    assert RE.md["versions"].get("bluesky") == bluesky.__version__


def test_ophydversion(RE):
    assert RE.md["versions"].get("ophyd") == ophyd.__version__


def test_eventmodelversion(RE):
    assert RE.md["versions"].get("event_model") == event_model.__version__


def test_old_md_validator(RE):
    """
    Test an old-style md_validator.

    In older versions of bluesky we ignored the result of md_validator.
    All it could do was raise; it could not provide a normalized valid
    result.

    When an md_validator returns None, we should use the input dict.
    """

    def md_validator(md):
        return None

    start_doc = None

    def test_callback(name, doc):
        nonlocal start_doc
        if name == "start":
            start_doc = doc

    RE.md_validator = md_validator
    RE(count([], md={"test": 1}), test_callback)
    assert start_doc["test"] == 1


def test_md_mormalizer(RE):
    """
    Test the new md_normalizer.

    In current versions of bluesky, md_normalizer is introduced to run
    alongside md_validator. It returns a normalized valid result,
    or raises if it is unable to provide one.

    When an md_normalizer returns a dict, we should use that.
    """

    def md_normalizer(md):
        "Ensure top-level keys are lowercase."
        return {key.lower(): value for key, value in md.items()}

    metadata = {"TEST": 1, "a": {"b": {"c": [1]}}}
    start_doc = None

    def test_callback(name, doc):
        nonlocal start_doc
        if name == "start":
            start_doc = doc

    RE.md_normalizer = md_normalizer
    RE(count([], md=metadata), test_callback)
    assert "TEST" in metadata
    assert "TEST" not in start_doc
    assert "test" in start_doc
    assert start_doc["test"] == 1
    assert start_doc["a"]["b"]["c"] is not metadata["a"]["b"]["c"]


class CustomMapping(MutableMapping):
    """A MutableMapping subclass to test that RE.md accepts non-dict mappings."""

    def __init__(self, *args, **kwargs):
        self._store = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)


def test_mutable_mapping_as_md(RE):
    """Test that RE.md works with a MutableMapping subclass instead of dict."""
    md = CustomMapping({"project": "test_project", "sample": "sample_1"})
    RE.md = md
    RE.md.setdefault("versions", {})

    assert RE.md["project"] == "test_project"
    assert RE.md["sample"] == "sample_1"

    start_doc = None

    def test_callback(name, doc):
        nonlocal start_doc
        if name == "start":
            start_doc = doc

    RE(count([]), test_callback)
    assert start_doc["project"] == "test_project"
    assert start_doc["sample"] == "sample_1"
