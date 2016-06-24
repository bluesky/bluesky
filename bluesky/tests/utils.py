from bluesky.run_engine import RunEngine
import contextlib
import tempfile
import sys


def setup_test_run_engine():
    # The metadata configured here used to be required for the RE to be
    # usable. Now it is all optional, but maintained for legacy reasons.
    RE = RunEngine()
    RE.md['owner'] = 'test_owner'
    RE.md['group'] = 'Grant No. 12345'
    RE.md['config'] = {'detector_model': 'XYZ', 'pixel_size': 10}
    RE.md['beamline_id'] = 'test_beamline'
    return RE


@contextlib.contextmanager
def _print_redirect():
    old_stdout = sys.stdout
    try:
        fout = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        sys.stdout = fout
        yield fout
    finally:
        sys.stdout = old_stdout


class MsgCollector:
    def __init__(self, msg_hook=None):
        self.msgs = []
        self.msg_hook = msg_hook

    def __call__(self, msg):
        self.msgs.append(msg)
        if self.msg_hook:
            self.msg_hook(msg)
