import asyncio
import os
import random
import signal
import time as ttime
from pprint import pprint
from bluesky import IllegalMessageSequence
from bluesky.tests.utils import setup_test_run_engine
from bluesky.utils import Msg, RunEngineInterrupted
import bluesky.examples as bse
from ophyd.sim import MockFlyer, SynGauss, Syn2DGauss, SynAxis
import uuid
import itertools
import gc
# random.seed(2016)


def unique_name():
    return str(uuid.uuid4())[:7]


# create objects
all_objects = set()


# create functions to get objects
def get_magic_flyer():
    """Get a new magic flyer. Magic flyers require no additional arguments in
    their message: Msg('kickoff', get_magic_flyer())
    """
    det1 = get_1d_det()
    motor = det1._motor
    flyname = 'mock_flyer_' + unique_name()
    flymagic = MockFlyer(flyname, motor, det1, -1, 1, 15)
    all_objects.add(flymagic)
    return flymagic


def get_flyer():
    """Create a new flyer object. Flyers require start, stop, steps in their
    message: Msg('kickoff', get_flyer(), start, stop, step)

    Returns
    -------
    MockFlyer object
    """
    det = get_1d_det()
    flyer = MockFlyer('wheee', det, det._motor, -1, 1, 15)
    all_objects.add(flyer)
    return flyer


def get_1d_det():
    """Create a new 1d detector

    Returns
    -------
    SynGauss object
    """
    motor = get_motor()
    detname = 'det1d_' + unique_name()
    det = SynGauss(detname, motor, motor.name, center=random.random() - 0.5,
                   Imax=random.random() * 10, sigma=random.random() * 5)
    all_objects.add(det)
    return det


def get_2d_det():
    """Create a new 2d detector

    Returns
    -------
    Syn2DGauss object
    """
    x = get_motor()
    y = get_motor()
    detname = 'det2d_' + unique_name()
    det = Syn2DGauss(detname, x, x.name, y, y.name,
                     center=(random.sample(range(-10, 10), 2)),
                     Imax=random.randrange(1, 10),
                     sigma=random.randrange(0, 5),
                     noise=random.choice(['poisson', 'uniform', None]),
                     noise_multiplier=random.random()*5)
    all_objects.add(det)
    return det


def get_motor():
    """Create a new motor object

    Returns
    -------
    Mover object
    """
    mtr_name = 'mtr_' + unique_name()
    mtr = SynAxis(name=mtr_name, delay=random.random() / 20)
    all_objects.add(mtr)
    return mtr


# get a list of messages
def get_messages(source_file):
    with open(source_file, 'r') as f:
        message_lines = set()
        for line in f.readlines():
            if 'Msg' in line and 'import' not in line:
                # remove everything before yield
                line = line.split('yield', 1)[-1].strip()
                # remove any line comments
                line = line.split('#')[0].strip()
                message_lines.add(line)
    # Turn the message list into actual message objects
    message_objects = []
    for line in message_lines:
        try:
            obj = eval(line)
        except Exception:
            print("Can't eval %s" % line)
        else:
            message_objects.append(obj)
    return message_objects


def interrupt_func():
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)
    ttime.sleep(0.5)


def kill_func():
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)
    ttime.sleep(0.1)
    os.kill(pid, signal.SIGINT)


def spam_SIGINT():
    pid = os.getpid()
    for _ in range(100):
        os.kill(pid, signal.SIGINT)
        print("sending SIGINT right now.")
        ttime.sleep(0.01)


def randomly_SIGINT_in_the_future():
    loop = asyncio.get_event_loop()
    for _ in range(10):
        func = interrupt_func
        if random.random() > 0.5:
            func = kill_func
        # randomly kill or interrupt at some point in the future. Oh, and
        # do it 10 times
        sigint_in_future = random.random() * 30
        print("SIGINT in {}".format(sigint_in_future))
        loop.call_later(sigint_in_future, func)


all_scan_generator_funcs = [
    bse.simple_scan,
    bse.conditional_break,
    bse.sleepy,
    bse.do_nothing,
    # checkpoint_forever should be part of these scans, as it will indefinitely
    # hang the fuzzing effort
    # checkpoint_forever,
    bse.wait_one,
    bse.wait_multiple,
    bse.wait_complex,
    bse.conditional_pause,
    bse.simple_scan_saving,
    bse.stepscan,
    bse.cautious_stepscan,
    bse.fly_gen,
    bse.multi_sample_temperature_ramp]


scan_kwarg_map = {
    'threshold': lambda: random.random() * 0.25 + 0.5,
    'start': lambda: random.randrange(-10, 0),
    'stop': lambda: random.randrange(0, 10),
    'step': lambda: random.randrange(5, 15),
    'motor': lambda: get_motor(),
    'det': lambda: get_1d_det() if random.random() < 0.5 else get_2d_det(),
    'timeout': lambda: random.random() * 3,
    'motors': lambda: [get_motor() for _ in range(random.randrange(1, 5))],
    'flyer': lambda: get_flyer(),
}


def make_scan_from_gen_func(scan):
    """Get a scan generator

    Parameters
    ----------
    scan : func
        The scan generator function to use to create a scan function.  The
        function signature will be inspected to determine the object types
        which need to be sent in with the function call.
        e.g., if the generator function is `def sleepy(det, motor):` then
        this function will see that there is a `det` field and a `motor` field
        and create a detector object and a motor object and then return
        `scan_gen(created_detector, created_motor)`

    Returns
    -------
    scan : generator
        A scan that should be passed to RunEngine.__call__
    """
    varnames = list(scan.__code__.co_varnames[:scan.__code__.co_argcount])
    kwargs = {k: scan_kwarg_map[k]() for k in varnames}
    if 'det' in varnames and 'motor' in varnames:
        det = scan_kwarg_map['det']()
        motor = det._motor
        kwargs['det'] = det
        kwargs['motor'] = motor
    return scan(**kwargs)


def get_scan_generators():
    """Get the list of scan generators that I can programmatically create

    Returns
    -------
    list
        The list of scan generators that I can inspect and create with my
        scan_kwarg_map

    """
    bad_scans = []
    for scan in all_scan_generator_funcs:
        varnames = scan.__code__.co_varnames[:scan.__code__.co_argcount]
        if not set(varnames).issubset(set(scan_kwarg_map)):
            bad_scans.append(scan)
    return [scan for scan in all_scan_generator_funcs if scan not in bad_scans]


def get_shuffleable_scan_generators():
    scans = get_scan_generators()
    RE = setup_test_run_engine()
    unshuffleable = []
    for scan in scans:
        print("Testing to see if {} is shufflable".format(
                scan.__code__.co_name))
        scan_gen = make_scan_from_gen_func(scan)
        # turn it into a list
        try:
            scan_list = list(scan_gen)
        except TypeError as te:
            print("{} is not shuffleable. Error: {}".format(
                    scan.__code__.co_name, te))
            unshuffleable.append(scan)
        # then shuffle it
        random.shuffle(scan_list)
        try:
            if RE.state != 'idle':
                RE.abort()
            RE(scan_list)
        except (IllegalMessageSequence, ValueError):
            # this is acceptable
            pass
    return [scan for scan in scans if scan not in unshuffleable]


def kickoff_and_collect(block=False, magic=False):
    """Make a flyer or magic_flyer object and maybe block

    Returns
    -------
    msg1 : Msg
        kickoff message for flyer or magic_flyer created in this function
    msg2 : Msg
        collect message for flyer or magic_flyer created in this function
    """
    args = []
    kwargs = {}
    if magic:
        flyer = get_magic_flyer()
    else:
        flyer = get_flyer()
        args = [random.randrange(-5, -1),  # start
                random.randrange(0, 5),    # stop
                random.randint(1, 10)]     # step

    if block:
        kwargs = {'group': unique_name()}
    return [Msg('kickoff', flyer, *args, **kwargs), Msg('collect', flyer)]


def run_fuzz():
    loop = asyncio.get_event_loop()
    # create 10 different flyers with corresponding kickoff and collect
    # messages
    flyer_messages = [msg for _ in range(10)
                      for msg in kickoff_and_collect(
                              block=random.random() > 0.5,
                              magic=random.random() > 0.5)]
    set_messages = [Msg('set', mtr, i) for i, mtr in
                    itertools.product(range(-5, 6),
                                      [get_motor() for _ in range(5)])]
    read_messages = [Msg('read', obj) for obj in all_objects
                     if hasattr(obj, 'read')]
    trigger_messages = [Msg('trigger', obj) for obj in all_objects
                        if hasattr(obj, 'trigger')]
    stage_messages = [Msg('stage', obj) for obj in all_objects]
    unstage_messages = [Msg('unstage', obj) for obj in all_objects]
    # throw random garbage at the open run metadata to see if we can break it
    openrun_garbage = [random.sample(gc.get_objects(), random.randint(1, 10))
                       for _ in range(10)]
    openrun_messages = [Msg('open_run', None, {str(v): v for v in garbage})
                        for garbage in openrun_garbage]
    closerun_messages = [Msg('close_run')] * 10
    checkpoint_messages = [Msg('checkpoint')] * 10
    clear_checkpoint_messages = [Msg('clear_checkpoint')] * 10
    create_messages = ([Msg('create')] * 5 +
                       [Msg('create', name=unique_name()) for _ in range(5)])
    save_messages = [Msg('save')] * 10
    sleep_messages = [Msg('sleep', None, random.random() * 0.25) for _ in range(10)]
    pause_messages = [Msg('pause')] * 10
    null_messages = [Msg('null')] * 10
    configure_messages = [Msg('configure', obj, d={}) for obj in all_objects
                          if hasattr(obj, 'configure')]

    # compile the list of all messages that we can send at the run engine
    message_objects = (flyer_messages + set_messages + read_messages +
                       trigger_messages + stage_messages + unstage_messages +
                       openrun_messages + closerun_messages +
                       checkpoint_messages + clear_checkpoint_messages +
                       create_messages + save_messages +
                       sleep_messages + pause_messages + null_messages +
                       configure_messages)
    print("Using the following messages")
    pprint(message_objects)

    RE = setup_test_run_engine()
    print("I am missing the following types of messages from my list")
    print(set(RE._command_registry.keys()) -
          set([msg.command for msg in message_objects]))

    num_message = 100
    num_SIGINT = 10
    num_scans = 50
    num_shuffled_scans = 50
    random.shuffle(message_objects)
    choices = (['message'] * num_message +
               # ['sigint'] * num_SIGINT +
               ['scan'] * num_scans +
               ['shuffled_scan'] * num_shuffled_scans)
    sigints = [
        ('interrupt', ['paused', 'idle'], interrupt_func),
        ('kill', ['idle'], kill_func),
        ('spam SIGINT', ['idle'], spam_SIGINT),
        ('random SIGINTs', ['idle', 'paused'], randomly_SIGINT_in_the_future)
    ]

    scan_generator_funcs = get_scan_generators()
    shufflable_scans = get_shuffleable_scan_generators()

    msg_seq = []
    count = 0
    while count < 500:
        name = random.choice(choices)
        if name == 'message':
            try:
                # grab a random sequence of messages
                msg = random.choice(message_objects)
                print(msg.command)
                msg_seq.append(msg.command)
                try:
                    RE([msg])
                except RunEngineInterrupted:
                    pass
                if msg.command == 'pause':
                    RE.resume()
                assert RE.state == 'idle'
            except IllegalMessageSequence as err:
                print(err)
        elif name == 'scan':
            scan_gen_func = random.choice(scan_generator_funcs)
            print("Running scan: {}"
                  "".format(scan_gen_func.__code__.co_name))
            scan_generator = make_scan_from_gen_func(scan_gen_func)
            try:
                RE(scan_generator)
            except RunEngineInterrupted:
                pass
        elif name == 'shuffled_scan':
            scan_gen_func = random.choice(shufflable_scans)
            print("Running shuffled scan: {}"
                  "".format(scan_gen_func.__code__.co_name))
            shuffled_scan = make_scan_from_gen_func(scan_gen_func)
            try:
                RE(shuffled_scan)
            except RunEngineInterrupted:
                pass
        elif name == 'sigint':
            sigint_type, allowed_states, func = random.choice(sigints)
            print(sigint_type)
            msg_seq.append(sigint_type)
            # TODO Figure out how to verify that the RunEngine is in
            # the desired state after this call_later executes
            loop.call_later(1, func)
        else:
            raise NotImplementedError("{} is not implemented".format(name))
        count += 1
        if count % 100 == 0:
            print('processed %s messages' % count)
        # Make sure the Run Engine is in a reasonable state
        if RE.state == 'idle':
            # all is well
            pass
        elif RE.state == 'running':
            # uh oh
            raise RuntimeError("Somehow the RunEngine thinks it is running")
            # RE.abort()
        elif RE.state == 'paused':
            RE.abort()
        else:
            # no way we can get here
            raise RuntimeError("How is the run engine even in this state? {}"
                               "".format(RE.state))
        try:
            # make sure we are idle before the next iteration
            assert RE.state == 'idle'
            # there is a chance that a sigint will get thrown between the end
            # of the above if/else block and this assert...
        except AssertionError:
            if RE.state == 'paused':
                RE.abort()
                assert RE.state == 'idle'
    print('%s actions were thrown at the RunEngine' % count)
    print("Fuzz testing completed successfully")
    print("Actions taken in the following order")
    pprint(msg_seq)
    print("Fuzz testing did not use the following messages")
    pprint(set(RE._command_registry.keys()) - set(msg_seq))


if __name__ == "__main__":
    run_fuzz()
