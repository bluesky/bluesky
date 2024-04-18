"""Extra utilities for state machines, to make them more usable."""

from weakref import WeakKeyDictionary


class ProxyString(str):

    """String that proxies every call to nested machine."""

    def __new__(cls, value, machine):
        """Create new string instance with reference to given machine."""
        string = super(cls, cls).__new__(cls, value)
        string.state_machine = machine
        return string

    def __getattr__(self, name):
        """Proxy call to machine."""
        return getattr(self.state_machine, name)


class PropertyMachine(object):

    """Descriptor to help using machines as properties."""

    def __init__(self, machine_type):
        """Create descriptor."""
        self.memory = WeakKeyDictionary()
        self.machine_type = machine_type

    def __set__(self, instance, value):
        """Set state to machine."""
        self.check_memory(instance)
        self.memory[instance].set_(value)

    def __get__(self, instance, _type=None):
        """Get machine state."""
        if instance is None:
            return self
        self.check_memory(instance)
        machine = self.memory[instance]
        return ProxyString(machine.actual_state.value, machine)

    def check_memory(self, instance):
        try:
            self.memory[instance]
        except KeyError:
            self.memory[instance] = self.machine_type()
