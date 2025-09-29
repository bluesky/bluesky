import json
import os

import pytest

from bluesky.callbacks.json_writer import JSONLinesWriter, JSONWriter


def read_json_file(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl_file(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("writer_class, extension", [(JSONWriter, "json"), (JSONLinesWriter, "jsonl")])
def test_json_writer(tmpdir, writer_class, extension):
    writer = writer_class(tmpdir)
    start_doc = {"uid": "abc", "value": 1}
    event_doc = {"seq_num": 1, "data": {"x": 1}}
    stop_doc = {"exit_status": "success"}

    writer("start", start_doc)
    writer("event", event_doc)
    writer("stop", stop_doc)

    # Read the file and check its contents
    filename = os.path.join(tmpdir, f"abc.{extension}")
    data = read_json_file(filename) if (writer_class == JSONWriter) else read_jsonl_file(filename)
    assert data[0]["name"] == "start"
    assert data[1]["name"] == "event"
    assert data[2]["name"] == "stop"


@pytest.mark.parametrize("writer_class, extension", [(JSONWriter, "json"), (JSONLinesWriter, "jsonl")])
def test_custom_filename(tmpdir, writer_class, extension):
    writer = writer_class(tmpdir, filename=f"custom.{extension}")
    doc = {"uid": "value"}
    writer("start", doc)
    assert os.path.exists(os.path.join(tmpdir, f"custom.{extension}"))
