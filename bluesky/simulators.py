from warnings import warn
from bluesky.preprocessors import print_summary_wrapper
from collections import namedtuple, defaultdict
import inspect

_TimeStats = namedtuple('TimeStats', 'est_time std_dev')
_MsgStats = namedtuple('MsgStats', 'message stop_time start_time')
_GroupStats = namedtuple('GroupStats', 'stop_time start_time')


def plot_raster_path(plan, x_motor, y_motor, ax=None, probe_size=None, lw=2):
    """Plot the raster path for this plan

    Parameters
    ----------
    plan : iterable
       Must yield `Msg` objects and not be a co-routine

    x_motor, y_motor : str
       Names of the x and y motors

    ax : matplotlib.axes.Axes
       The axes to plot to, if none, make new figure + axes

    probe_size : float, optional
       If not None, use as radius of probe (in same units as motor positions)

    lw : float, optional
        Width of lines drawn between points
    """
    import matplotlib.pyplot as plt
    from matplotlib import collections as mcollections
    from matplotlib import patches as mpatches
    if ax is None:
        ax = plt.subplots()[1]
    ax.set_aspect('equal')

    cur_x = cur_y = None
    traj = []
    for msg in plan:
        cmd = msg.command
        if cmd == 'set':
            if msg.obj.name == x_motor:
                cur_x = msg.args[0]
            if msg.obj.name == y_motor:
                cur_y = msg.args[0]
        elif cmd == 'save':
            traj.append((cur_x, cur_y))

    x, y = zip(*traj)
    path, = ax.plot(x, y, marker='', linestyle='-', lw=lw)
    ax.set_xlabel(x_motor)
    ax.set_ylabel(y_motor)
    if probe_size is None:
        read_points = ax.scatter(x, y, marker='o', lw=lw)
    else:
        circles = [mpatches.Circle((_x, _y), probe_size,
                                   facecolor='black', alpha=0.5)
                   for _x, _y in traj]

        read_points = mcollections.PatchCollection(circles,
                                                   match_original=True)
        ax.add_collection(read_points)
    return {'path': path, 'events': read_points}


def summarize_plan(plan):
    """Print summary of plan

    Prints a minimal version of the plan, showing only moves and
    where events are created.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects
    """
    for msg in print_summary_wrapper(plan):
        ...



print_summary = summarize_plan  # back-compat


class LimitsExceeded(Exception):
    ...


def check_limits(plan):
    """
    Check that a plan will not move devices outside of their limits.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects

    Raises
    ------
    LimitsExceeded
    """
    ignore = []
    for msg in plan:
        if msg.command == 'set':
            if msg.obj in ignore:
                continue  # we have already warned about this device
            try:
                low, high = msg.obj.limits
            except AttributeError:
                warn("Limits of {} are unknown and can't be checked.".format(msg.obj.name))
                ignore.append(msg.obj)
                continue
            if low == high:
                warn("Limits are not set on {}".format(msg.obj.name))
                ignore.append(msg.obj)
                continue
            pos, = msg.args
            if not (low < pos < high):
                raise LimitsExceeded("This plan would put {} at {} "
                                     "which is outside of its limits, {}."
                                     "".format(msg.obj.name, pos, (low, high)))


"""
ATTRIBUTE PARAMETERS
--------------------
_run_start_cmds: list
    A list of message 'commands' that are found at the start of a run.
_run_end_cmds: list
    A list of message 'commands' that are found at the end of a run.
_group_start_cmds: list
    A list of message 'commands' that are found at the begining of a 'group'
    (where more than one command is processed in parallel).
_group_end_cmds: list
    A list of message 'commands' that are found at the begining of a 'group'
    (where more than one command is processed in parallel).
_self._timed_cmds: list.
    A list of message 'commands' that do not start/end a run/group/flyer but
    are stil timed.
__flyer_start_cmds: list.
    A list of message 'commands' that start a 'flyer'.
_plan_history: dict.
    A dictionary containing information on values updated during the plan. It
    has key:arg pairs with keys relating to message components where each arg
    is a dictionary containing devices (detectors, axes) that have been
    updated, as keywords and a value indicating the update status. For 'set'
    the update status is the latest 'position' that it is to be set and for
    'trigger' the update status is the number of times since the beginning of
    the scan or since an 'unstage' event that the device has been triggered.
"""
# GROUP THE DIFFERENT COMMAND TYPES
# The commands that start/end a 'group', but not flyer commands.
_group_start_cmds = ['set', 'trigger', 'kickoff']
_group_end_cmds = ['wait']
# The commands that aren't  groupable but still have statistics.
_timed_cmds = ['stage', 'unstage']
# The commands that only require network communication.
_communication_cmds = ['read', 'describe', 'open_run', 'close_run']
# Define some times associated with network communications and internal
# state tasks. I am not yet sure how to accurately estimate these times as
# they are unlikely to be 'static' and measuring them for statistics will
# take as much time as performing them, they are including incase a good
# solution for this is found.
_communication_time = _TimeStats(0, 0)  # An est_time/std_dev named tuple
_internal_state_time = _TimeStats(0, 0)  # An est_time/std_dev named tuple


def est_time(plan):
    '''The generator function estimates the time to complete a plan with
    a yield at every message.

    This method takes in a plan and yields out a tuple with the structure
    (msg, estimated stop time, estimated start time). Where msg is the
    message from the plan and stop/start times are (est_time, std_dev)
    named tuples which provides the end/start time for each message taking
    the scan start as zero. Note that for 'groupable' actions (like 'set'
    and 'trigger') the action msg will start and stop within the group time.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the est_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan,
        the est_time/ std_dev named tuple for cumulative time and the est_time
        /std_dev named tuple for the message time.
    '''

    # create a reference to a dictionary called plan_history to capture set
    # changes during the plan for later modification.
    plan_history = {'set': {}, 'trigger': {}}
    # create a dictionary for storing the info regarding groups in progress.
    groups = defaultdict(_TimeStats())

    # used to hold the cumulative time for the plan
    out_time = _TimeStats(0, 0)

    for msg in plan:
        # case 1: commands that are 'groupable'
        if msg.command in _group_start_cmds:
            # add set and trigger commands to plan_history
            if msg.command == 'set':
                plan_history['set'][msg.obj.name] = msg.args[0]
            elif msg.command == 'trigger':
                if msg.obj.name in list(plan_history['trigger'].keys()):
                    plan_history['trigger'][msg.obj.name] += 1
                else:
                    plan_history['trigger'][msg.obj.name] = 1

            message_time = combine_est(
                            obj_est(msg, plan_history=plan_history),
                            _communication_time)
            start_time = out_time
            stop_time = combine_est(start_time, message_time)

            try:
                group_time = groups[msg.kwargs.get('group')]
            except KeyError:
                group_time = _GroupStats(_TimeStats(0, 0), _TimeStats(0, 0))

            # update the times for the group.
            groups[msg.kwargs.get('group')] = _GroupStats(combine_est(
                                                 group_time.stop_time,
                                                 stop_time,
                                                 method='max'),
                                                 combine_est(
                                                 group_time.start_time,
                                                 start_time,
                                                 method='min'))
            # yield out the times for the message
            yield _MsgStats(msg, stop_time, start_time)

        # case 2: commands that end 'groups'
        elif msg.command in _group_end_cmds:
            group_time = groups.pop(msg.kwargs.get('group'))

            start_time = group_time.start_time
            out_time = combine_est(out_time, group_time.stop_time)

            yield _MsgStats(msg, out_time, start_time)

        # case 3: commands that have stats but are not 'groupable'
        elif msg.command in _timed_cmds:
            start_time = out_time
            out_time = combine_est(out_time,
                                   obj_est(msg, plan_history=plan_history))

            yield _MsgStats(msg, out_time, start_time)

        # case 4: commands that have an associated communication time
        elif msg.command in _communication_cmds:
            start_time = out_time
            out_time = combine_est(out_time, _communication_time)

            yield _MsgStats(msg, out_time, start_time)

        # case 5: sleep command
        elif msg.command is 'sleep':
            start_time = out_time
            out_time = combine_est(out_time, _TimeStats(msg.args[0], 0))
            out_time = combine_est(out_time, _communication_time)

            yield _MsgStats(msg, out_time, start_time)

        # case 6: commands that only require internal processing
        else:
            start_time = out_time
            out_time = combine_est(out_time, _internal_state_time)

            yield _MsgStats(msg, out_time, _communication_time)


def combine_est(est_time_1, est_time_2, method='sum'):
    """
    Returns the combination est_time/std_dev pairs et_1 and et_2.
    This function returns the combination of est_time_1 and est_time_2,
    combined using the method definedby 'method'.

    Parameters
    ----------
    est_time_1, est_time_2 : namedtuples.
        The tuples containing the est_time/std_dev tuples to be combined.
    method : string, optional.
        The method to use for the combination of est_time_1 and est_time_2,
        default is 'sum'.

    Return Parameters
    -----------------
    out_time : namedtuple.
        The combined est_time/std_dev tuple.
    """

    if method == 'sum':
        # adds the est_time, uses quadrature for std_dev.
        out_time = _TimeStats(est_time_1.est_time + est_time_2.est_time,
                              (est_time_1.std_dev**2 + est_time_2.std_dev**2)
                              ** .5)

    if method == 'subtract':
        # adds the est_time, uses quadrature for std_dev.
        out_time = _TimeStats(est_time_1.est_time - est_time_2.est_time,
                              (est_time_1.std_dev**2 + est_time_2.std_dev**2)
                              ** .5)

    elif method == 'max':
        # finds the max est_time, finds the potentially largest difference from
        # the max_time as the std_dev.
        max_time = max(est_time_1.est_time, est_time_2.est_time)
        pos_stdev = max(est_time_1.est_time + est_time_1.std_dev,
                        est_time_2.est_time + est_time_2.std_dev)
        neg_stdev = min(est_time_1.est_time - est_time_1.std_dev,
                        est_time_2.est_time - est_time_2.std_dev)
        max_stdev = max(abs(pos_stdev-max_time), abs(neg_stdev-max_time))
        out_time = _TimeStats(max_time, max_stdev)

    elif method == 'min':
        # finds the min est_time, finds the potentially largest difference from
        # the min_time as the std_dev.
        min_time = min(est_time_1.est_time, est_time_2.est_time)
        pos_stdev = max(est_time_1.est_time + est_time_1.std_dev,
                        est_time_2.est_time + est_time_2.std_dev)
        neg_stdev = min(est_time_1.est_time - est_time_1.std_dev,
                        est_time_2.est_time - est_time_2.std_dev)
        max_stdev = max(abs(pos_stdev-min_time), abs(neg_stdev-min_time))
        out_time = _TimeStats(min_time, max_stdev)

    return out_time


def obj_est(msg, plan_history={}):
    """
    Returns the est_time/std_dev pair for the object referenced in msg.
    This function returns the est_time/std_dev pair for the object referenced
    in msg.obj and for the command type reference in msg.command.

    Parameters
    ----------
    msg : message.
        The first message of the group that was detected in plan_ETA.
    plan_history : dict, optional.
        A dictionary containing information on values updated during the plan.
        It has key:arg pairs with keys relating to message components where
        each arg is a dictionary containing devices (detectors, axes) that
        have been updated, as keywords and a value indicating the update
        status. For 'set' the update status is the latest 'position' that it
        is to be set and for 'trigger' the update status is the number of
        times since the beginning of the scan or since an 'unstage' event that
        the device has been triggered.

    Return Parameters
    -----------------
    out_time : named tuple.
        The combined est_time/std_dev pair.
       """

    set_dict = {}

    if msg.obj is not None:
        obj = msg.obj
        if msg.command == 'set':
            try:
                set_dict['start_pos'] = plan_history['set'][msg.obj.name]
            except KeyError:
                set_dict['start_pos'] = msg.obj.position

            set_dict['target'] = msg.args[0]

        # determine, and find, what values are required for time estimation
        obj_est_time = getattr(msg.obj.est_time, msg.command)
        arg_names = inspect.signature(obj_est_time).parameters
        args = []
        for arg_name in arg_names:
            try:
                args.append(set_dict[arg_name])
            except KeyError:
                try:
                    args.append(plan_history['set'][arg_name])
                except KeyError:
                    try:
                        args.append(getattr(obj, arg_name).position)
                    except AttributeError:
                        print('{}.est_time.{} requires a {} attribute on \
                              {} but none can be found'.format(obj.name,
                              msg.command, arg_name, obj))
                        raise

        # ask the object for a time estimation.
        object_est = obj_est_time(*args)

        return object_est
    else:
        return _TimeStats(0, 0)


def est_time_per_group(plan):
    '''The generator function removes any yields from est_time(plan) that
    occur within a group.

    This method yields out a tuple with the structure (msg, estimated stop
    time, estimated start time). Where msg is the message from the plan and
    stop/start times are (est_time, std_dev) named tuples which provides the
    end/start time for each message taking the scan start as zero. It removes
    any yields from 'groupable' messages but does yield for the group as a
    whole, at the group end message.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan,
        the est_time/ std_dev named tuple for cumulative time and the est_time
        /std_dev named tuple for the message time.
    '''

    for estimated_time in est_time(plan):
        if estimated_time.message.command not in _group_start_cmds:
            yield estimated_time


def est_time_run(plan):
    '''The generator function adds the start/stop times for 'runs' to
    est_time(plan) these will be yielded at the 'close_run'.

    This method yields out a tuple with the structure (msg, estimated stop
    time, estimated start time). Where msg is the message from the plan and
    stop/start times are (est_time, std_dev) named tuples which provides the
    end/start time for each message taking the scan start as zero. It removes
    any yields from 'groupable' messages but does yield for the group as a
    whole, at the group end message.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan,
        the est_time/ std_dev named tuple for cumulative time and the est_time
        /std_dev named tuple for the message time.
    '''
    # Note: when parallel runs are allowed this will need updating.
    for estimated_time in est_time(plan):
        if estimated_time.message.command is 'open_run':
            start_time = estimated_time.start_time
            yield estimated_time
        elif estimated_time.message.command is 'close_run':
            stop_time = estimated_time.stop_time
            yield _MsgStats(estimated_time.message, stop_time, start_time)
        else:
            yield estimated_time


def est_time_per_run(plan):
    '''The generator function removes any yields from est_time(plan) that
    occur within a group.

    This method yields out a tuple with the structure (msg, estimated stop
    time, estimated start time). Where msg is the message from the plan and
    stop/start times are (est_time, std_dev) named tuples which provides the
    end/start time for each message taking the scan start as zero. It removes
    any yields from 'groupable' messages but does yield for the group as a
    whole, at the group end message.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan,
        the est_time/ std_dev named tuple for cumulative time and the est_time
        /std_dev named tuple for the message time.
    '''

    for estimated_time in est_time_run(plan):
        if estimated_time.message.command == 'open_run':
            inner_estimated_time = estimated_time
            while inner_estimated_time.message.command is not 'close_run':
                inner_estimated_time = next(est_time_run(plan))

            yield inner_estimated_time
        else:
            yield estimated_time


def print_est_time(plan, est_time_func=est_time_per_run):
    '''This function prints to the command line the estimated start and stop
    times provided by est_time_func(plan).

    This function prints to the command line the estimated start/stop times,
    and their associated standard deviations, for each message returned by
    est_time_func. Thereby providing the information at the verbosity level
    of est_time_func.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be printed for.

    est_time_func : generator.
        The time estimation generator to use, which determines how much
        information is provided.
    '''

    run_id = -1  # This will not be used once runs can occur in parallel

    for estimated_time in est_time_func(plan):
        time_string = f'   ->  start: \
                      {estimated_time.start_time.est_time}+/-\
                      {estimated_time.start_time.std_dev},  stop: \
                      {estimated_time.stop_time.est_time}+/-\
                      {estimated_time.stop_time.std_dev}'

        if estimated_time.command == 'set':
            print(f'move {estimated_time.message.obj.name} to \
                  {estimated_time.message.args[0]}' + time_string)

        elif estimated_time.command == 'sleep':
            print(f'wait for {estimated_time.message.args[0]}' + time_string)

        elif estimated_time.command == 'close_run':
            print(f'close_run {run_id}' + time_string)

        elif estimated_time.command == 'wait':
            print(f'wait on group \
                  {estimated_time.message.kwargs.get("group")}' +
                  time_string)

        else:
            print(f'{estimated_time.message.command} \
                    {estimated_time.message.obj.name}' + time_string)
