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
    return


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
    mtr = Mover(mtr_name, [mtr_name], sleep_time=random.random())
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
    ttime.sleep(0.05)
    os.kill(pid, signal.SIGINT)


def spam_sigint():
    pid = os.getpid()
    for _ in range(100):
        os.kill(pid, signal.SIGINT)
        ttime.sleep(0.005)

# define the random message generation functions
def kickoff_and_collect(block=False, magic=False):
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
    from bluesky import examples
    RE = setup_test_run_engine()
    source_file = examples.__file__
    print("Using the following message objects")

    # create 10 different flyers with corresponding kickoff and collect
    # messages
    flyer_messages = [msg for _ in range(10)
                      for msg in kickoff_and_collect(block=random.random()>0.5,
                                                     magic=random.random()>0.5)]
    set_messages = [Msg('set', motor, i) for i, motor in
                    itertools.product(range(-5, 6),
                                      [get_motor() for _ in range(5)])]
    read_messages = [Msg('read', obj) for obj in all_objects]
    trigger_messages = [Msg('trigger', obj) for obj in all_objects
                        if hasattr(obj, 'trigger')]
    stage_messages = [Msg('stage', obj) for obj in all_objects]
    unstage_messages = [Msg('unstage', obj) for obj in all_objects]
    # add some messages to set a motor to random positions
    message_objects = (flyer_messages + set_messages + read_messages +
                       trigger_messages + stage_messages + unstage_messages)
    print("Using the following messages")
    for obj in message_objects:
        print(obj)

    print("I am missing the following types of messages from my list")
    print(set(RE._command_registry.keys()) -
          set([msg.command for msg in message_objects]))

    num_message = 100
    num_interrupt = 2
    num_kill = 2
    num_hammer_ctrl_c = 2
    message = ('message', lambda: random.choice(message_objects), ['idle'])
    interrupt = ('interrupt', interrupt_func, ['paused', 'idle'])
    kill = ('kill', kill_func, ['idle'])
    hammer_ctrl_c = ('spam ctrl+c', spam_sigint, ['idle'])

    actions = ([message] * num_message +
               [interrupt] * num_interrupt +
               [kill] * num_kill +
               [hammer_ctrl_c] * num_hammer_ctrl_c)

    msg_seq = []
    count = 0
    while count < 250:
        name, action, expected_state = random.choice(actions)
        if name == 'message':
            try:
                msg = action()
                msg_seq.append(msg)
                RE([msg])
            except IllegalMessageSequence as err:
                print(err)
            # except Exception as e:
            #     pdb.set_trace()
        else:
            msg_seq.append(name)
        try:
            assert RE.state in expected_state
        except AssertionError:
            pdb.set_trace()
        count += 1
        if count % 100 == 0:
            print('processed %s messages' % count)

    print("Fuzz testing completed successfully")
    print("Actions taken in the following order")
    pprint(msg_seq)


if __name__ == "__main__":
    run_fuzz()