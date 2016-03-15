import os
import random
import signal
import time as ttime
import pdb
from pprint import pprint
from bluesky import IllegalMessageSequence
from bluesky.examples import *
from bluesky.tests.utils import setup_test_run_engine
import uuid
import itertools
import gc
random.seed(2016)

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
    det2 = get_1d_det()
    det2._motor = motor
    det2._motor_field = motor.name
    scan_points = random.choice(range(15))
    flyname = 'flymagic_' + unique_name()
    flymagic = FlyMagic(flyname, motor, det1, det2, scan_points=scan_points)
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
    flyer = MockFlyer(det, det._motor)
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
    det = SynGauss(detname, motor, motor.name, center=random.randrange(-1, 1),
                   Imax=random.randrange(1, 10), sigma=random.randrange(0, 5))
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
    det = Syn2DGauss(detname, x, x.name, y, x.name,
                     center=(random.sample(range(-10, 10), 2)),
                     Imax=random.randrange(1, 10),
                     sigma=random.randrange(0, 5),
                     noise=random.choice(['poission', 'uniform', None]),
                     noise_multiplier=random.random(0, 5))
    all_objects.add(det)
    return det


def get_motor():
    """Create a new motor object

    Returns
    -------
    Mover object
    """
    mtr_name = 'mtr_' + unique_name()
    mtr = Mover(mtr_name, [mtr_name], sleep_time=random.random() / 20)
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
         ttime.sleep(0.01)
         print("siginting")


def randomly_SIGINT_in_the_future():
    for _ in range(10):
        func = interrupt_func
        if random.random() > 0.5:
            func = kill_func
        # randomly kill or interrupt at some point in the future. Oh, and
        # do it 10 times
        loop.call_later(random.random() * 30, func)


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
        kwargs = {'block_group': unique_name()}
    return [Msg('kickoff', flyer, *args, **kwargs), Msg('collect', flyer)]


def run_fuzz():
    # create 10 different flyers with corresponding kickoff and collect
    # messages
    flyer_messages = [msg for _ in range(10)
                      for msg in kickoff_and_collect(block=random.random()>0.5,
                                                     magic=random.random()>0.5)]
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
    num_SIGINT = 8
    random.shuffle(message_objects)
    choices = (['message'] * num_message +
               ['sigint'] * num_SIGINT)
    sigints = [
        ('interrupt', ['paused', 'idle'], interrupt_func),
        ('kill', ['idle'], kill_func),
        ('spam SIGINT', ['idle'], spam_SIGINT),
        ('random SIGINTs', ['idle', 'paused'], randomly_SIGINT_in_the_future)
    ]


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
                RE([msg])
                assert RE.state == 'idle'
            except IllegalMessageSequence as err:
                print(err)
        elif name == 'sigint':
            sigint_type, allowed_states, func = random.choice(sigints)
            print(sigint_type)
            msg_seq.append(sigint_type)
            #TODO Figure out how to verify that the RunEngine is in the desired state after this call_later executes
            loop.call_later(1, func)
        count += 1
        if count % 100 == 0:
            print('processed %s messages' % count)
        # Make sure the Run Engine is in a reasonable state
        if RE.state == 'idle':
            # all is well
            pass
        elif RE.state == 'running':
            # uh oh
            RE.abort()
        elif RE.state == 'paused':
            RE.abort()
        else:
            # no way we can get here
            raise RuntimeError("How is the run engine even in this state? {}"
                               "".format(RE.state))
        # make sure we are idle before the next iteration
        assert RE.state == 'idle'
    print('%s actions were thrown at the RunEngine' % count)
    print("Fuzz testing completed successfully")
    print("Actions taken in the following order")
    pprint(msg_seq)
    print("Fuzz testing did not use the following messages")
    pprint(set(RE._command_registry.keys()) - set(msg_seq))


if __name__ == "__main__":
    run_fuzz()
