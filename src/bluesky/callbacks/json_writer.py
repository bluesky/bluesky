import json
from pathlib import Path


class JSONWriter:
    """Writer of Bluesky docuemnts into a JSON file

    The file is written immediately, and the JSON array is closed
    when the "stop" document is received.
    If `jsonlines` is set to True, each document is written as a separate line
    in the file, without enclosing it in an array.
    """

    def __init__(self, dirname: str, filename: str = None, jsonlines: bool = False):
        self.dirname = Path(dirname)
        self.filename = filename
        self.jsonlines = jsonlines

    def __call__(self, name, doc):
        if name == "start":
            self.filename = self.filename or f"{doc['uid']}.{'jsonl' if self.jsonlines else 'json'}"
            mode = "a" if (self.dirname / self.filename).exists() else "w"
            with open(self.dirname / self.filename, mode) as file:
                if not self.jsonlines:
                    file.write("[\n")
                json.dump({"name": name, "doc": doc}, file)

        elif name == "stop":
            with open(self.dirname / self.filename, "a") as file:
                if not self.jsonlines:
                    file.write(",")
                file.write("\n")
                json.dump({"name": name, "doc": doc}, file)
                if not self.jsonlines:
                    file.write("\n]")

        else:
            with open(self.dirname / self.filename, "a") as file:
                if not self.jsonlines:
                    file.write(",")
                file.write("\n")
                json.dump({"name": name, "doc": doc}, file)


class DelayedJSONWriter:
    """Writer of Bluesky documents into a JSON file.

    The file is written only after (and if) flush is called.
    """

    def __init__(self, dirname, filename=None, jsonlines=False):
        self.dirname = Path(dirname)
        self.filename = filename
        self.jsonlines = jsonlines
        self._docs = []

    def __call__(self, name, doc):
        self._docs.append({"name": name, "doc": doc})
        if name == "start" and not self.filename:
            self.filename = f"{doc['uid']}.{'jsonl' if self.jsonlines else 'json'}"

    def flush(self):
        mode = "a" if (self.dirname / self.filename).exists() else "w"
        with open(self.dirname / self.filename, mode) as file:
            if self.jsonlines:
                for item in self._docs:
                    json.dump(item, file)
                    file.write("\n")
            else:
                json.dump(self._docs, file, indent=2)
        self._docs.clear()
