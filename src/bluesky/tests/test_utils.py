import pytest
import numpy as np

from functools import reduce
import operator

from bluesky.utils import (
    ensure_generator, Msg, merge_cycler, is_movable, CallbackRegistry,
    warn_if_msg_args_or_kwargs
)
from cycler import cycler


def test_single_msg_to_gen():
    m = Msg('set', None, 0)

    m_list = [m for m in ensure_generator(m)]

    assert len(m_list) == 1
    assert m_list[0] == m


@pytest.mark.parametrize('traj',
                         ({'pseudo1': [10, 11, 12],
                           'pseudo2': [20, 21, 22]},
                          {'pseudo1': [10, 11, 12],
                           'pseudo2': [20, 21, 22],
                           'pseudo3': [30, 31, 32]},
                          )
                         )
def test_cycler_merge_pseudo(hw, traj):
    p3x3 = hw.pseudo3x3
    sig = hw.sig
    keys = traj.keys()
    tlen = len(next(iter(traj.values())))
    expected_merge = [{k: traj[k][j] for k in keys}
                      for j in range(tlen)]

    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), v)
                                for k, v in traj.items()))

    mcyc = merge_cycler(cyc + cycler(sig, range(tlen)))

    assert mcyc.keys == {p3x3, sig}
    assert mcyc.by_key()[p3x3] == expected_merge
    assert mcyc.by_key()[sig] == list(range(tlen))


@pytest.mark.parametrize('children',
                         (
                             ['pseudo1', 'real1'],
                             ['pseudo1', 'pseudo2', 'real1']))
def test_cycler_merge_pseudo_real_clash(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))

    with pytest.raises(ValueError):
        merge_cycler(cyc)


@pytest.mark.parametrize('children',
                         (['pseudo1', ],
                          ['pseudo1', 'pseudo2'],
                          ['real1'],
                          ['real1', 'real2']))
def test_cycler_parent_and_parts_fail(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))
    cyc += cycler(p3x3, range(5))

    with pytest.raises(ValueError):
        merge_cycler(cyc)


@pytest.mark.parametrize('children',
                         (['sig', ],))
def test_cycler_parent_and_parts_succed(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))
    cyc += cycler(p3x3, range(5))
    mcyc = merge_cycler(cyc)

    assert mcyc.keys == cyc.keys
    assert mcyc.by_key() == cyc.by_key()


@pytest.mark.parametrize('children',
                         (
                             ['pseudo1'],
                             ['pseudo2', 'sig'],
                             ['real1'],
                             ['real1', 'real2'],
                             ['real1', 'real2', 'sig'],
                         ))
def test_cycler_merge_mixed(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))

    mcyc = merge_cycler(cyc)

    assert mcyc.keys == cyc.keys
    assert mcyc.by_key() == cyc.by_key()


def test_is_movable(hw):

    obj_list = [(10, False), (1.05, False), ("some_string", False),
                (hw.det, False), (hw.motor, True)]
    for obj, result in obj_list:
        assert is_movable(obj) == result, \
            f"The object {obj} is incorrectly recognized "\
            f"as {'' if result else 'not '}movable"


# Indicates if the step when all external references to callables are deleted is included.
@pytest.mark.parametrize("delete_objects", [True, False])
# Settings for the 'set_allowed_signals' parameter of the CallbackRegistry class.
@pytest.mark.parametrize("set_allowed_signals", [True, False])
# Callable type
@pytest.mark.parametrize("callable_type", ["bound_method",
                                           "function",
                                           "callable_object",
                                           "class_method",
                                           "static_method",
                                           "callable_object_class_method",
                                           "callable_object_static_method"])
def test_CallbackRegistry_1(delete_objects, set_allowed_signals, callable_type):
    """
    Basic tests for CallbackRegistry class. The tests verify the behavior
    of CallbackRegistry when connecting and disconnecting callback functions
    represented as the following types:
      - bound methods,
      - functions,
      - callable objects,
      - class methods,
      - static methods.
    The tests performed on callable objects with __call__ function being class or
    static method. The class needs to be instantiated: `a = A()` and the object `a`
    is used as a reference to the callback in the call to subscribe. (I don't know if anyone
    would do such a thing.)

    The callable objects are clearly separated into two groups:
      - GROUP 1 - callable objects;
      - GROUP 2 - functions, bound methods, class methods and static methods.

    The following behavior is expected when connecting callbacks to Callback Registry:

    GROUP 1 (bound methods) - weak reference is held internally (using proxy), deleting external
    references invalidates the weak reference and the respective callback is removed
    from the registry. Callback can be removed manually in a standard way by using `disconnect`
    function.

    GROUP 2 (the rest) - strong reference to the callable object is held internally,
    deleting all external references does not influence the callbacks in CB registry,
    callback can be used until they are removed by 'disconnect' function.

    The dictionaries of Callback registry: callbacks, _func_cid_map. Both dictionaries
    are using signal names as keys. In the current implementation, the entries in the two dictionaries
    are deleted if all callbacks for the respective signals are automatically disconnected
    (callbacks are bound methods and external references are deleted for a bound method).
    If all callbacks (actually the last callback) for a signal are manually disconnected,
    then the respective items in the dictionaries will remain in both dictionaries (they will
    hold ZERO callback references).
    """
    if set_allowed_signals:
        allowed_sigs = {"sig1", "sig2", "sig3"}
    else:
        allowed_sigs = None

    cb = CallbackRegistry(allowed_sigs=allowed_sigs)

    def _f_print(fn, kwarg_value):
        """Formatting of output"""
        return f"Function {fn}: kwarg_value {kwarg_value}"

    signals = {"sig1": 5, "sig2": 4, "sig3": 3}  # sig_name: num_of_created_objects

    # Lists to store data on the created objects
    obj_to_delete = []  # List of objects that will be explicitly deleted in the test
    obj_cid = []  # Object CID (returned by 'CallbackRegistyr.connect()' method
    obj_name = []  # Object Name (assigned to the object in order to identify it in the output
    obj_signal = []  # Name of the signal to which the object with respective index is subscribed.

    # Create objects of the selected type
    for sig_name, n_objects in signals.items():
        for _ in range(n_objects):
            n_callable = len(obj_to_delete)  # The index of the current callable

            if callable_type == "function":

                def _get_f(*, func_name):
                    def f(list_out, *, kwarg_value):
                        list_out.append(_f_print(f"{func_name}", kwarg_value))
                    return f

                o_name = f"f{n_callable}"
                f = _get_f(func_name=o_name)
                o_subscribe = f
                o_delete = f
                del f

            elif callable_type == "callable_object":
                def _get_class_instance(*, func_name):
                    class cl:
                        def __init__(self, func_name):
                            self._func_name = func_name

                        def __call__(self, list_out, *, kwarg_value):
                            list_out.append(_f_print(f"{self._func_name}", kwarg_value))

                    return cl(func_name)

                o_name = f"f{n_callable}"
                cl = _get_class_instance(func_name=o_name)
                o_subscribe = cl
                o_delete = cl
                del cl

            elif callable_type == "bound_method":
                def _get_class_instance(*, func_name):
                    class cl:
                        def __init__(self, func_name):
                            self._func_name = func_name

                        def func(self, list_out, *, kwarg_value):
                            list_out.append(_f_print(f"{self._func_name}", kwarg_value))

                    return cl(func_name)

                o_name = f"f{n_callable}"
                cl_inst = _get_class_instance(func_name=o_name)
                o_subscribe = cl_inst.func
                o_delete = cl_inst
                del cl_inst

            elif callable_type == "class_method":
                def _get_class_instance(*, func_name):
                    class cl:
                        @classmethod
                        def func(cls, list_out, *, kwarg_value):
                            list_out.append(_f_print(f"{func_name}", kwarg_value))

                    return cl()

                o_name = f"f{n_callable}"
                cl = _get_class_instance(func_name=o_name)
                o_subscribe = cl.func
                o_delete = cl
                del cl

            elif callable_type == "static_method":
                def _get_class_instance(*, func_name):
                    class cl:
                        @staticmethod
                        def func(list_out, *, kwarg_value):
                            list_out.append(_f_print(f"{func_name}", kwarg_value))

                    return cl()

                o_name = f"f{n_callable}"
                cl = _get_class_instance(func_name=o_name)
                o_subscribe = cl.func
                o_delete = cl
                del cl

            elif callable_type == "callable_object_class_method":
                def _get_class_instance(*, func_name):
                    class cl:
                        @classmethod
                        def __call__(cls, list_out, *, kwarg_value):
                            list_out.append(_f_print(f"{func_name}", kwarg_value))

                    return cl()

                o_name = f"f{n_callable}"
                cl_inst = _get_class_instance(func_name=o_name)
                o_subscribe = cl_inst
                o_delete = cl_inst
                del cl_inst

            elif callable_type == "callable_object_static_method":
                def _get_class_instance(*, func_name):
                    class cl:
                        @staticmethod
                        def __call__(list_out, *, kwarg_value):
                            list_out.append(_f_print(f"{func_name}", kwarg_value))

                    return cl()

                o_name = f"f{n_callable}"
                cl_inst = _get_class_instance(func_name=o_name)
                o_subscribe = cl_inst
                o_delete = cl_inst
                del cl_inst

            else:
                raise RuntimeError(f"Unknown type of the callable: {callable_type}")

            obj_to_delete.append(o_delete)
            obj_signal.append(sig_name)
            obj_name.append(o_name)
            cid = cb.connect(sig_name, o_subscribe)
            obj_cid.append(cid)
            del o_subscribe, o_delete

    # Verify that the right number of callbacks was initially set
    assert len(cb._func_cid_map) == len(signals), "Incorrect number of signals"
    assert len(cb.callbacks) == len(signals), "Incorrect number of signals"
    for sig_name, n_objects in signals.items():
        assert len(cb._func_cid_map[sig_name]) == n_objects, \
            f"Incorrect number of callbacks for '{sig_name}'"
        assert len(cb.callbacks[sig_name]) == n_objects, \
            f"Incorrect number of callbacks for '{sig_name}'"

    def _process_each_signal(n_start_check=0):
        """Process each signal, check callback output starting from index `n_start_check`"""
        # Try calling each signal
        for sig_name in signals.keys():
            # The list of indices for the entries related to 'signal_name' signal
            i_sig = [_ for _ in range(len(obj_signal)) if (obj_signal[_] == sig_name)]

            list_out = []
            rand_value = np.random.rand()  # Some value that is expected to be part of the function output
            cb.process(sig_name, list_out, kwarg_value=rand_value)

            assert len(list_out) == len([_ for _ in i_sig if _ >= n_start_check]), \
                "Output list has incorrect number of entries"
            for n in i_sig:
                if n >= n_start_check:
                    expected_substr = _f_print(obj_name[n], rand_value)
                    assert list_out.count(expected_substr) == 1, \
                        f"Signal '{sig_name}' was processed incorrectly: entry '{expected_substr}' " \
                        f"was not found in the output '{list_out}'"

    _process_each_signal()

    if delete_objects:
        # Now delete all the callable objects one by one
        for n in range(len(obj_to_delete)):
            obj_to_delete[n] = None  # Overwriting the reference deletes the object

            # Check the function composition
            if callable_type in ["function", "callable_object", "class_method", "static_method",
                                 "callable_object_class_method", "callable_object_static_method"]:
                # Deleting objects should change nothing
                assert len(cb._func_cid_map) == len(signals), "Incorrect number of signals"
                assert len(cb.callbacks) == len(signals), "Incorrect number of signals"
                for sig_name, n_objects in signals.items():
                    assert len(cb._func_cid_map[sig_name]) == n_objects, \
                        f"Incorrect number of callbacks for '{sig_name}'"
                    assert len(cb.callbacks[sig_name]) == n_objects, \
                        f"Incorrect number of callbacks for '{sig_name}'"

                _process_each_signal()

            elif callable_type == "bound_method":
                # Callbacks should be removed as they get deleted
                sigs_remaining = list(set(obj_signal[n+1:]))
                assert len(cb._func_cid_map) == len(sigs_remaining), "Incorrect number of signals"
                assert len(cb.callbacks) == len(sigs_remaining), "Incorrect number of signals"
                for sig_name, n_objects in signals.items():
                    if sig_name in sigs_remaining:
                        assert len(cb._func_cid_map[sig_name]) == obj_signal[n+1:].count(sig_name), \
                            f"Incorrect number of callbacks for '{sig_name}'"
                        assert len(cb.callbacks[sig_name]) == obj_signal[n+1:].count(sig_name), \
                            f"Incorrect number of callbacks for '{sig_name}'"

                _process_each_signal(n_start_check=n + 1)

            else:
                assert False, f"Unknown callable type: {callable_type}"

    if delete_objects and callable_type == "bound_method":
        # Dictionary entries for signals are deleted when the objects are deleted
        #   and callbacks are unsubscribed
        assert len(cb._func_cid_map) == 0, "Not all callbacks were automatically unsubscribed"
        assert len(cb.callbacks) == 0, "Not all callbacks were automatically unsubscribed"
    else:
        # Now disconnect the callbacks one by one and verify dictionary content
        for n in range(len(obj_to_delete)):
            cb.disconnect(obj_cid[n])
            # NOTE: as the objects are deleted, the dictionary entries for the signals still remain
            assert len(cb._func_cid_map) == len(signals), "Incorrect number of signals"
            assert len(cb.callbacks) == len(signals), "Incorrect number of signals"
            for sig_name, n_objects in signals.items():
                assert len(cb._func_cid_map[sig_name]) == obj_signal[n+1:].count(sig_name), \
                    f"Incorrect number of callbacks for '{sig_name}'"
                assert len(cb.callbacks[sig_name]) == obj_signal[n+1:].count(sig_name), \
                    f"Incorrect number of callbacks for '{sig_name}'"
            _process_each_signal(n_start_check=n + 1)


def test_CallbackRegistry_2():
    """
    The following features were tested:
    - connect the same callback to the same signal (repeated connections are ignored);
    - connecting the same callback to different signals (works noramlly);
    - connecting a callback to a signals that are not in allowed list (exception raised);
    - attempting to process a signals that are not in allowed list (exception raised;
    - processing signals that have no callback assigned (nothing happens).
    """

    allowed_sigs = {"sig1", "sig2", "sig3"}
    cb = CallbackRegistry(allowed_sigs=allowed_sigs)

    def f():
        pass

    # Signal not allowed
    with pytest.raises(ValueError, match=f"Allowed signals are {allowed_sigs}"):
        cb.connect("some_sig", f)

    # Connect callback to allowed signal
    sig_name = "sig2"
    cb.connect(sig_name, f)
    assert len(cb.callbacks[sig_name]) == 1, "Incorrect number of assigned callbacks"

    # Attempt to connect the same callback to the same signal are ignored
    cb.connect(sig_name, f)
    assert len(cb.callbacks[sig_name]) == 1, "Incorrect number of assigned callbacks"

    # Connect the same callback to a different signal
    sig_name2 = "sig3"
    cb.connect(sig_name2, f)
    assert len(cb.callbacks[sig_name]) == 1, "Incorrect number of assigned callbacks"
    assert len(cb.callbacks[sig_name2]) == 1, "Incorrect number of assigned callbacks"

    # Process the allowed signal with assigned callback
    cb.process(sig_name)

    # Process the allowed signal with unassigned callback (no callbacks called,
    #   but it still works.
    cb.process("sig1")

    # Process the signal that is not allowed
    with pytest.raises(ValueError, match=f"Allowed signals are {allowed_sigs}"):
        cb.process("some_signal")


def test_msg_args_kwargs_emits_warning_first_time(recwarn):
    class MyDevice:
        def kickoff(self, *args, **kwargs):
            pass

    device = MyDevice()
    msg = Msg("kickoff")

    assert len(recwarn) == 0
    warn_if_msg_args_or_kwargs(msg, device.kickoff, (), {"arg": "value"})
    assert len(recwarn) == 1
    w = recwarn.pop(PendingDeprecationWarning)
    assert str(w.message) == """\
About to call kickoff() with args () and kwargs {'arg': 'value'}.
In the future the passing of Msg.args and Msg.kwargs down to hardware from
Msg("kickoff") may be deprecated. If you have a use case for these,
we would like to know about it, so please open an issue at
https://github.com/bluesky/bluesky/issues"""
    assert len(recwarn) == 0
    # Second time doesn't warn
    warn_if_msg_args_or_kwargs(msg, device.kickoff, (), {"arg": "value"})
    assert len(recwarn) == 0
    # Called without kwargs doesn't warn
    warn_if_msg_args_or_kwargs(msg, device.kickoff, (), {})
    assert len(recwarn) == 0
