from bluesky.examples import *
from bluesky.tests.utils import setup_test_run_engine
import random
from bluesky import IllegalMessageSequence

motors = [motor, motor1, motor2, motor3]
dets = [det, det1, det2, det3, noisy_det]

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



def run_fuzz():
    from bluesky import examples
    RE = setup_test_run_engine()
    source_file = examples.__file__
    print("Using the following message objects")

    message_objects = get_messages(source_file)
    # add some messages to set a motor to random positions
    message_objects.extend([Msg('set', motor, i) for i in range(-5, 6)])
    # add some configure messages
    message_objects.extend([Msg('configure', obj, {}) for obj in motors+dets])
    for obj in message_objects:
        print(obj)

    print("I am missing the following types of messages from my list")
    print(set(RE._command_registry.keys()) -
          set([msg.command for msg in message_objects]))

    msg_seq = []
    count = 0
    while count < 100:
        msg = random.choice(message_objects)
        msg_seq.append(msg)
        try:
            RE([msg])
        except IllegalMessageSequence as err:
            print(err)
        assert RE.state == 'idle'
        count += 1
        if count % 100 == 0:
            print('processed %s messages' % count)

    print("Fuzz testing completed successfully")



if __name__ == "__main__":
    run_fuzz()