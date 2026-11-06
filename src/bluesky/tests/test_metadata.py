import asyncio
from collections.abc import MutableMapping
from typing import Any

import event_model
import ophyd
import pytest

import bluesky
import bluesky.plan_stubs as bps
from bluesky import Msg, RunEngineInterrupted
from bluesky.plans import count
from bluesky.run_engine import RunEngine
from bluesky.utils import MsgGenerator


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


def test_context_md_defaults_to_empty_dict(RE: RunEngine) -> None:
    assert RE.context_md == {}


def test_context_md_constructor_arg() -> None:
    context_md = CustomMapping({"calib": "abc"})
    loop = asyncio.new_event_loop()
    RE = RunEngine({}, context_md=context_md, loop=loop)
    try:
        assert RE.context_md is context_md
    finally:
        loop.call_soon_threadsafe(loop.stop)
        RE._th.join()
        loop.close()


@pytest.mark.parametrize("context", [False, True])
def test_set_and_get_metadata(RE: RunEngine, context: bool) -> None:
    result: Any = None

    def plan() -> MsgGenerator[None]:
        nonlocal result
        yield from bps.set_metadata(a=1, b=[2, 3], context=context)
        result = yield from bps.get_metadata("a", "b", context=context)

    RE(plan())
    assert result == (1, [2, 3])

    target, other = (RE.context_md, RE.md) if context else (RE.md, RE.context_md)
    assert target["a"] == 1
    assert target["b"] == [2, 3]
    assert "a" not in other
    assert "b" not in other


def test_get_metadata_single_key_returns_value(RE: RunEngine) -> None:
    RE.md["sample"] = "Si"
    RE.context_md["sample"] = "Ge"
    result: dict[str, Any] = {}

    def plan() -> MsgGenerator[None]:
        result["md"] = yield from bps.get_metadata("sample")
        result["context_md"] = yield from bps.get_metadata("sample", context=True)

    RE(plan())
    assert result == {"md": "Si", "context_md": "Ge"}


def test_set_metadata_overwrites_existing(RE: RunEngine) -> None:
    RE.md["sample"] = "Si"
    RE(bps.set_metadata(sample="Ge"))
    assert RE.md["sample"] == "Ge"


def test_get_metadata_missing_key_raises(RE: RunEngine) -> None:
    with pytest.raises(KeyError, match="missing"):
        RE(bps.get_metadata("missing"))


@pytest.mark.parametrize("context", [False, True])
def test_del_metadata(RE: RunEngine, context: bool) -> None:
    target, other = (RE.context_md, RE.md) if context else (RE.md, RE.context_md)
    target.update(a=1, b=2, c=3)
    other["a"] = 1

    RE(bps.del_metadata("a", "b", context=context))

    assert "a" not in target
    assert "b" not in target
    assert target["c"] == 3
    assert other["a"] == 1


def test_del_metadata_missing_key_ignored(RE: RunEngine) -> None:
    RE.md["a"] = 1
    RE(bps.del_metadata("missing", "a"))
    assert "a" not in RE.md


def test_del_metadata_replayed_on_rewind(RE: RunEngine) -> None:
    commands: list[str] = []
    RE.msg_hook = lambda msg: commands.append(msg.command)
    RE.md["sample"] = "Si"

    def plan() -> MsgGenerator[None]:
        yield from bps.checkpoint()
        yield from bps.del_metadata("sample")
        yield from bps.pause()

    with pytest.raises(RunEngineInterrupted):
        RE(plan())
    assert "sample" not in RE.md

    RE.md["sample"] = "Ge"
    RE.resume()

    assert "sample" not in RE.md
    assert commands.count("del_metadata") == 2


def test_set_metadata_in_plan_is_included_in_next_start_doc(RE: RunEngine) -> None:
    start_docs: list[dict[str, Any]] = []

    def plan() -> MsgGenerator[None]:
        yield from bps.set_metadata(sample="Si")
        yield from bps.set_metadata(calib="xyz", context=True)
        yield from count([])

    RE(plan(), lambda name, doc: start_docs.append(doc) if name == "start" else None)
    assert len(start_docs) == 1
    assert start_docs[0]["sample"] == "Si"
    assert "calib" not in start_docs[0]


def test_set_metadata_persists_across_plans(RE: RunEngine) -> None:
    RE(bps.set_metadata(operator="jdoe", context=True))
    result: Any = None

    def plan() -> MsgGenerator[None]:
        nonlocal result
        result = yield from bps.get_metadata("operator", context=True)

    RE(plan())
    assert result == "jdoe"


def test_metadata_msgs_with_raw_msg(RE: RunEngine) -> None:
    result: Any = None

    def plan() -> MsgGenerator[None]:
        nonlocal result
        yield Msg("set_metadata", None, {"x": 5})
        result = yield Msg("get_metadata", None, "x")

    RE(plan())
    assert result == 5


def _pausing_metadata_plan(results: list[Any], context: bool) -> MsgGenerator[None]:
    yield from bps.checkpoint()
    yield from bps.set_metadata(sample="Si", context=context)
    results.append((yield from bps.get_metadata("sample", context=context)))
    yield from bps.pause()
    results.append((yield from bps.get_metadata("sample", context=context)))


@pytest.mark.parametrize("context", [False, True])
def test_metadata_msgs_replayed_on_rewind(RE: RunEngine, context: bool) -> None:
    store: MutableMapping[str, Any] = RE.context_md if context else RE.md
    commands: list[str] = []
    RE.msg_hook = lambda msg: commands.append(msg.command)
    results: list[Any] = []

    with pytest.raises(RunEngineInterrupted):
        RE(_pausing_metadata_plan(results, context))
    assert store["sample"] == "Si"
    assert results == ["Si"]

    # Resuming rewinds to the checkpoint, so the cached set_metadata clobbers this edit.
    store["sample"] = "Ge"
    RE.resume()

    assert store["sample"] == "Si"
    # The replayed get_metadata response is not sent back into the plan.
    assert results == ["Si", "Si"]
    assert commands.count("set_metadata") == 2
    assert commands.count("get_metadata") == 3


@pytest.mark.parametrize("context", [False, True])
def test_metadata_msgs_not_replayed_when_not_rewindable(RE: RunEngine, context: bool) -> None:
    store: MutableMapping[str, Any] = RE.context_md if context else RE.md
    commands: list[str] = []
    RE.msg_hook = lambda msg: commands.append(msg.command)
    RE.rewindable = False
    results: list[Any] = []

    with pytest.raises(RunEngineInterrupted):
        RE(_pausing_metadata_plan(results, context))
    assert store["sample"] == "Si"

    store["sample"] = "Ge"
    RE.resume()

    assert store["sample"] == "Ge"
    assert results == ["Si", "Ge"]
    assert commands.count("set_metadata") == 1
    assert commands.count("get_metadata") == 2
