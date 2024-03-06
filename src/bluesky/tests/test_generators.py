import uuid
import pytest
from collections import deque
from itertools import zip_longest

from bluesky import Msg

from bluesky.preprocessors import (msg_mutator, stub_wrapper,
                                   plan_mutator, pchain, single_gen as
                                   single_message_gen,
                                   finalize_wrapper)

from bluesky.utils import ensure_generator


class EchoException(Exception):
    ...


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
            if msg.command == 'FAIL':
                plan.throw(EchoException(msg))
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


def test_mutator_exceptions():
    handled = False

    def base_plan():
        yield Msg('foo')
        yield Msg('bar')

    def failing_plan():
        nonlocal handled
        handled = False
        yield Msg('pre')
        try:
            yield Msg('FAIL')
        except EchoException:
            handled = True
            raise

    def test_mutator(msg):
        if msg.command == 'bar':
            return (
                failing_plan(),
                single_message_gen(Msg('foo'))
            )
        return None, None

    # check generator exit behavior
    plan = plan_mutator(base_plan(), test_mutator)
    next(plan)
    plan.close()

    # check exception fall through
    plan = plan_mutator(base_plan(), test_mutator)
    with pytest.raises(EchoException):
        EchoRE(plan, debug=True)
    assert handled


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
    plan = finalize_wrapper(erroring_plan(),
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


def test_finialize_pause():
    fail_cmd = 'fail_next'

    def erroring_plan():
        yield Msg(fail_cmd, None)
        raise RuntimeError('saw this coming')

    num = 5
    cmd = 'echo'
    plan = finalize_wrapper(erroring_plan(),
                            echo_plan(command=cmd, num=num),
                            pause_for_debug=True)
    msgs = list()
    try:
        EchoRE(plan, msg_list=msgs)
    except RuntimeError:
        pass

    total = num + 2
    _verify_msg_seq(msgs, m_len=total,
                    cmd_sq=[fail_cmd, 'pause'] + [cmd]*num,
                    args_sq=[()]*total,
                    kwargs_sq=[{}, {'defer': False}] + [{}]*num)


def test_finialize_success():
    suc_cmd = 'it_works'

    num = 5
    cmd = 'echo'
    plan = finalize_wrapper(single_message_gen(Msg(suc_cmd, None)),
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


def test_exception_in_pre_with_tail():
    class SnowFlake(Exception):
        ...

    def bad_pre():
        yield Msg('pre_bad', None)
        raise SnowFlake('this one')

    def good_post():
        yield Msg('good_post', None)

    def test_mutator(msg):
        if msg.command == 'TARGET':
            return bad_pre(), good_post()

        return None, None

    def testing_plan():
        yield Msg('a', None)
        yield Msg('b', None)
        try:
            yield Msg('TARGET', None)
        except SnowFlake:
            pass
        yield Msg('b', None)
        yield Msg('a', None)

    plan = plan_mutator(testing_plan(), test_mutator)
    msgs = EchoRE(plan, debug=True)
    _verify_msg_seq(msgs, m_len=5,
                    cmd_sq=['a', 'b', 'pre_bad', 'b', 'a'],
                    args_sq=[()]*5,
                    kwargs_sq=[{}]*5)


def test_plan_mutator_returns():

    def testing_plan():
        yield Msg('a', None)
        yield Msg('TARGET', None)
        yield Msg('b', None)

        return 'foobar'

    def outer_plan(pln):
        ret = (yield from pln)
        assert ret == 'foobar'
        return ret

    def tail_plan():
        yield Msg('A', None)
        return 'baz'

    def test_mutator(msg):
        def pre_plan():
            yield Msg('pre', None)
            yield msg

        if msg.command == 'TARGET':
            return pre_plan(), tail_plan()

        return None, None

    plan = plan_mutator(testing_plan(), test_mutator)
    msgs = EchoRE(plan)
    _verify_msg_seq(msgs, m_len=5,
                    cmd_sq=['a', 'pre', 'TARGET', 'A', 'b'],
                    args_sq=[()]*5,
                    kwargs_sq=[{}]*5)


def test_insert_before():

    def target():
        yield Msg('a', None)
        ret = yield Msg('TARGET', None)
        yield Msg('b', None)
        assert ret.command == 'TARGET'
        return ret
        return ret

    def insert_before(msg):
        if msg.command == 'TARGET':
            def pre():
                yield Msg('pre', None)
                ret = yield msg
                assert ret is not None
                assert ret.command == 'TARGET'
                return ret

            return pre(), None
        else:
            return None, None

    EchoRE(plan_mutator(target(), insert_before))


def test_insert_after():

    def target():
        yield Msg('a', None)
        ret = yield Msg('TARGET', None)
        yield Msg('b', None)
        assert ret is not None
        assert ret.command == 'TARGET'
        return ret

    def insert_after(msg):
        if msg.command == 'TARGET':
            def post():
                yield Msg('post', None)

            return None, post()
        else:
            return None, None

    EchoRE(plan_mutator(target(), insert_after))


def test_base_exception():
    class SnowFlake(Exception):
        ...

    def null_mutator(msg):
        return None, None

    def test_plan():
        yield Msg('a', None)
        raise SnowFlake('this one')

    pln = plan_mutator(test_plan(), null_mutator)

    try:
        EchoRE(pln)
    except SnowFlake as ex:
        assert ex.args[0] == 'this one'


def test_msg_mutator_skip():
    def skipper(msg):
        if msg.command == 'SKIP':
            return None
        return msg

    def skip_plan():
        for c in 'abcd':
            yield Msg(c, None)
            yield Msg('SKIP', None)

    pln = msg_mutator(skip_plan(), skipper)
    msgs = EchoRE(pln)
    _verify_msg_seq(msgs, m_len=4,
                    cmd_sq='abcd',
                    args_sq=[()]*4,
                    kwargs_sq=[{}]*4)


def test_stub_wrapper():
    def plan():
        yield Msg('open_run')
        yield Msg('stage')
        yield Msg('read')
        yield Msg('unstage')
        yield Msg('close_run')

    stub_plan = list(stub_wrapper(plan()))
    assert len(stub_plan) == 1
    assert stub_plan[0].command == 'read'
