"""Tests for code."""

import os
import re

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
source_dir = os.path.join(root_dir, "super_state_machine")
tests_dir = os.path.join(root_dir, "tests")


def _collect_recursively(directory, result):
    for name in os.listdir(directory):
        if not re.search("^\\.", name):
            fullpath = os.path.join(directory, name)
            if re.search("\\.py$", name):
                result.append(fullpath)
            elif os.path.isdir(fullpath):
                _collect_recursively(fullpath, result)


def _collect_static(dirs):
    matches = []
    for dir_ in dirs:
        _collect_recursively(dir_, matches)
    return matches
