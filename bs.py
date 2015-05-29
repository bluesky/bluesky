import time as ttime
from itertools import count
from collections import namedtuple, deque


class Msg(namedtuple('Msg_base', ['message', 'obj', 'args', 'kwargs'])):
    __slots__ = ()

    def __repr__(self):
        return '{}: ({}), {}, {}'.format(
            self.message, self.obj, self.args, self.kwargs)


class Base:
    def __init__(self, name, fields):
        self._name = name
        self._fields = fields

    def describe(self):
        return {k: {'source': self._name, 'dtype': 'number'}
                for k in self._fields}

    def __repr__(self):
        return '{}: {}'.format(self._klass, self._name)


class Reader(Base):
    _klass = 'reader'
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
    _klass = 'mover'
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


def MoveRead_gen(motor, detector):
    try:
        for j in range(10):
            yield Msg('set', motor, ({'x': j}, ), {})
            yield Msg('trigger', motor, (), {})
            yield Msg('trigger', detector, (), {})
            yield Msg('read', detector, (), {})
            yield Msg('read', motor, (), {})
    finally:
        print('Generator finished')


class RunEngine:
    def __init__(self):
        self.panic = False
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
                if self.panic():
                    break
                msg = g.send(r)
                r = self._proc_registry[msg.message](msg)

                print('{}\n   ret: {}'.format(msg, r))
            except StopIteration:
                break

    def _create(self, msg):
        pass

    def _read(self, msg):
        return msg.obj.read(*msg.args, **msg.kwargs)

    def _save(self, msg):
        raise NotImplementedError()

    def _null(self, msg):
        pass

    def _set(self, msg):
        return msg.obj.set(*msg.args, **msg.kwargs)

    def _trigger(self, msg):
        return msg.obj.trigger(*msg.args, **msg.kwargs)

    def _wait(self, msg):
        return ttime.sleep(*msg.arg)


RE = RunEngine()

g = MoveRead_gen(Mover('motor', 'x'), Reader('det', ['cnt', ]))


RE.run_engine(g)
