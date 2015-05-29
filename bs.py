import time as ttime
from itertools import count
from collections import namedtuple, deque

class Base:
    def __init__(self, name, fields):
        self._name = name
        self._fields = fields

    def describe(self):
        return {k: {'source': self._name, 'dtype': 'number'}
                for k in self._fields}

class Reader(Base):
    def __init__(self, *args, **kwargs):
        super(Reader, self).__init__(*args, **kwargs)
        self._cnt = 0

    def read(self):
        data = dict()
        for k in self._fields:
            data[k] = self._cnt
            self._cnt += 1

        return data

    def trigger(self, *, blocking=True):
        if blocking:
            pass


class Mover(Base):
    def __init__(self, *args, **kwargs):
        super(Mover, self).__init__(*args, **kwargs)
        self._data = {k: 0 for k in self._fields}
        self._staging = None

    def read(self):
        return dict(self._data)

    def set(self, new_values):
        self._staging = new_values

    def trigger(self, *, blocking=True):
        if blocking:
            pass

        if self._staging:
            for k, v in self._staging.items():
                self._data[k] = v

        self._staging = None

    def settle(self):
        pass


class FlyMagic(Base):
    def kickoff(self):
        pass
    def collect(self):
        pass





def test_gen():
    try:
        for j in count():
            val = yield j
            if val:
                break
            yield j - .5
    finally:
        print('Generator finished')


class RunEngine:
    def __init__(self):
        self._read_cache = deque()
        self._proc_registry = {
            'create': self._create,
            'save': self._save,
            'read': self._read,
            'null': self._null,
            'set': self._set,
            'trigger': self._trigger,
            'wait': self._wait
            }

    def run_engine(self, g):
        r = None
        while True:
            try:
                msg = g.send(r)
                r = self._proc_registry[msg.message](msg)

                print(msg, r)
            except StopIteration:
                break

    def check_external(self):
        pass

    def _create(self, msg):
        pass

    def _read(self, msg):
        return msg.obj.read(*msg.args, **msg.kwargs)

    def _save(self, msg):
        raise NotImplementedError()

    def _null(self, msg):
        pass

    def _set(self, msg):
        raise NotImplementedError()

    def _trigger(self, msg):
        raise NotImplementedError()

    def _wait(self, msg):
        return time.sleep(*msg.arg)


RE = RunEngine()

g = test_gen()

RE.run_engine(g)
