import uuid
from collections import deque
import types
from itertools import zip_longest

from bluesky import Msg

from bluesky.plans import (msg_mutator, plan_mutator, pchain,
                           single_gen as single_message_gen, finalize)


def ensure_generator(plan):
    gen = iter(plan)  # no-op on generators; needed for classes
    if not isinstance(gen, types.GeneratorType):
        # If plan does not support .send, we must wrap it in a generator.
        gen = (msg for msg in gen)

    return gen


def EchoRE(plan, *, debug=False, msg_list=None):
    '''An 'echo' RunEngine for testing.

    Always sends the message back into the plan as the result.

    Parameters
    ----------
    plan : iterable
        The plan to run through

    debug : bool, optional (False)
        print the messages on the way by

    msg_list : mutable sequence, optional
        If not None, mutate this object by appending messages.
        This is the easiest way to capture messages if the plan
        raises.

    Returns
    -------
    msg_list : list
        List of all the messages seen by the RE
    '''
    if msg_list is None:
        msg_list = deque()
    ret = None
    plan = ensure_generator(plan)
    while True:
        try:
            msg = plan.send(ret)
            if debug:
                print(msg)
            msg_list.append(msg)
            ret = msg
        except StopIteration:
            break

    return list(msg_list)


def echo_plan(*, command='echo', num=4):
    '''Testing plan which expects to get back a message with an equal object
    '''
    seed = str(uuid.uuid4())[:6]
    for ch in map(lambda x: chr(97 + x), range(num)):
        sent = '{}_{}'.format(seed, ch)
        ret = yield Msg(command, sent)
        assert ret.obj == sent


def _verify_msg_seq(msgs, *,
                    cmd_sq=None,
                    obj_sq=None,
                    args_sq=None,
                    kwargs_sq=None,
                    m_len=None):

    def _verify_cmpt(msgs, seq, cmpt):
        for m, s in zip_longest(msgs, seq):
            assert getattr(m, cmpt) == s

    if m_len is not None:
        assert len(msgs) == m_len

    if cmd_sq is not None:
        _verify_cmpt(msgs, cmd_sq, 'command')

    if obj_sq is not None:
        _verify_cmpt(msgs, obj_sq, 'obj')

    if args_sq is not None:
        _verify_cmpt(msgs, args_sq, 'args')

    if kwargs_sq is not None:
        _verify_cmpt(msgs, kwargs_sq, 'kwargs')


def test_smoke_test():
    num = 10
    cmd = 'smoke'
    msgs = EchoRE(echo_plan(command=cmd, num=num))
    _verify_msg_seq(msgs, m_len=num,
                    cmd_sq=[cmd]*num,
                    args_sq=[()]*num,
                    kwargs_sq=[{}]*num)


def test_simple_replace():
    new_cmd = 'replaced'

    def change_command(msg):
        return msg._replace(command=new_cmd)

    num = 10
    msgs = EchoRE(msg_mutator(echo_plan(num=num),
                              change_command))
    _verify_msg_seq(msgs, m_len=num,
                    cmd_sq=[new_cmd]*num,
                    args_sq=[()]*num,
                    kwargs_sq=[{}]*num)


def test_simple_mutator():
    _mut_active = True
    pre_count = 3
    post_count = 5
    pre_cmd = 'pre'
    post_cmd = 'post'

    def test_mutator(msg):
        nonlocal _mut_active
        if _mut_active:
            _mut_active = False

            return (pchain(echo_plan(num=pre_count, command=pre_cmd),
                            single_message_gen(msg)),
                    echo_plan(num=post_count, command=post_cmd))
        return None, None

    num = 5
    cmd = 'echo'
    plan = plan_mutator(echo_plan(command=cmd, num=num), test_mutator)
    msgs = EchoRE(plan)
    total = num + pre_count + post_count
    cmd_sq = ([pre_cmd]*pre_count +
              [cmd] +
              [post_cmd]*post_count +
              [cmd]*(num-1))
    _verify_msg_seq(msgs, m_len=total,
                    cmd_sq=cmd_sq,
                    args_sq=[()]*total,
                    kwargs_sq=[{}]*total)


def test_finialize_fail():
    fail_cmd = 'fail_next'

    def erroring_plan():
        yield Msg(fail_cmd, None)
        raise RuntimeError('saw this coming')

    num = 5
    cmd = 'echo'
    plan = finalize(erroring_plan(),
                    echo_plan(command=cmd, num=num))
    msgs = list()
    try:
        EchoRE(plan, msg_list=msgs)
    except RuntimeError:
        pass

    total = num + 1
    _verify_msg_seq(msgs, m_len=total,
                    cmd_sq=[fail_cmd] + [cmd]*num,
                    args_sq=[()]*total,
                    kwargs_sq=[{}]*total)


def test_finialize_success():
    suc_cmd = 'it_works'

    num = 5
    cmd = 'echo'
    plan = finalize(single_message_gen(Msg(suc_cmd, None)),
                    echo_plan(command=cmd, num=num))
    msgs = list()
    try:
        EchoRE(plan, msg_list=msgs)
    except RuntimeError:
        pass

    total = num + 1
    _verify_msg_seq(msgs, m_len=total,
                    cmd_sq=[suc_cmd] + [cmd]*num,
                    args_sq=[()]*total,
                    kwargs_sq=[{}]*total)


def test_plan_mutator_exception_propogation():
    class ExpectedException(Exception):
        pass

    num = 5
    cmd1 = 'echo1'
    cmd2 = 'echo2'

    def bad_tail():
        yield Msg('one_tail', None)
        raise ExpectedException('this is a test')

    def sarfing_plan():
        try:
            yield from echo_plan(command=cmd1, num=num)
        except ExpectedException:
            print('CAUGHT IT')

    _mut_active = True

    def test_mutator(msg):
        nonlocal _mut_active
        if _mut_active:
            _mut_active = False

            return (pchain(echo_plan(num=2, command=cmd2),
                            single_message_gen(msg)),
                    bad_tail())
        return None, None

    plan = plan_mutator(sarfing_plan(), test_mutator)
    EchoRE(plan, debug=True)
