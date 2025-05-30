import json
from pathlib import Path


class JSONWriter:
    """Writer of Bluesky docuemnts of a single run into a JSON file as an array.

    The file is created when a Start doocument is received, each new document is
    written immediately, and the JSON array is closed when the "stop" document
    is received.
    """

    def __init__(
        self,
        dirname: str,
        filename: str = None,
    ):
        self.dirname = Path(dirname)
        self.filename = filename

    def __call__(self, name, doc):
        if name == "start":
            self.filename = self.filename or f"{doc['uid']}.json"
            with open(self.dirname / self.filename, "w") as file:
                file.write("[\n")
                json.dump({"name": name, "doc": doc}, file)
                file.write(",\n")

        elif name == "stop":
            with open(self.dirname / self.filename, "a") as file:
                json.dump({"name": name, "doc": doc}, file)
                file.write("\n]")

        else:
            with open(self.dirname / self.filename, "a") as file:
                json.dump({"name": name, "doc": doc}, file)
                file.write(",\n")


class JSONLinesWriter:
    """Writer of Bluesky docuemnts into a JSON Lines file

    The file is written immediately, and the JSON array is closed
    when the "stop" document is received.
    """

    def __init__(self, dirname: str, filename: str = None):
        self.dirname = Path(dirname)
        self.filename = filename

    def __call__(self, name, doc):
        if name == "start":
            self.filename = self.filename or f"{doc['uid']}.jsonl"
            mode = "a" if (self.dirname / self.filename).exists() else "w"
            with open(self.dirname / self.filename, mode) as file:
                json.dump({"name": name, "doc": doc}, file)
                file.write("\n")

        elif name == "stop":
            with open(self.dirname / self.filename, "a") as file:
                json.dump({"name": name, "doc": doc}, file)

        else:
            with open(self.dirname / self.filename, "a") as file:
                json.dump({"name": name, "doc": doc}, file)
                file.write("\n")
