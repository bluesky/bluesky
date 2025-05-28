import json
import os

import pytest

from bluesky.callbacks.json_writer import DelayedJSONWriter, JSONWriter


def read_json_file(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl_file(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("jsonlines", [True, False])
@pytest.mark.parametrize("writer_class", [JSONWriter, DelayedJSONWriter])
def test_json_writer(tmpdir, writer_class, jsonlines):
    writer = writer_class(tmpdir, jsonlines=jsonlines)
    start_doc = {"uid": "abc", "value": 1}
    event_doc = {"seq_num": 1, "data": {"x": 1}}
    stop_doc = {"exit_status": "success"}

    writer("start", start_doc)
    writer("event", event_doc)
    writer("stop", stop_doc)
    if isinstance(writer, DelayedJSONWriter):
        writer.flush()

    # Read the file and check its contents
    filename = os.path.join(tmpdir, "abc.jsonl" if jsonlines else "abc.json")
    data = read_jsonl_file(filename) if jsonlines else read_json_file(filename)
    assert data[0]["name"] == "start"
    assert data[1]["name"] == "event"
    assert data[2]["name"] == "stop"


@pytest.mark.parametrize("writer_class", [JSONWriter, DelayedJSONWriter])
def test_custom_filename(writer_class, tmpdir):
    writer = writer_class(tmpdir, filename="custom.json")
    doc = {"uid": "value"}
    writer("start", doc)
    if isinstance(writer, DelayedJSONWriter):
        writer.flush()
    assert os.path.exists(os.path.join(tmpdir, "custom.json"))
