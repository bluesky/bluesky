from warnings import warn
from bluesky.preprocessors import print_summary_wrapper
from collections import namedtuple, defaultdict
import inspect

_TimeStats = namedtuple('TimeStats', 'est_time std_dev')
_CumulativeStats = namedtuple('CumulativeStats',
                              'message stop_time start_time')
_MsgStats = namedtuple('MsgStats', 'message process_time action_time')
_GroupStats = namedtuple('GroupStats', 'delta_time elapsed_time')


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
                warn("Limits of {} are unknown and can't be checked.\
                     ".format(msg.obj.name))
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
_group_start_cmds: list
    A list of message 'commands' that are found at the begining of a 'group'
    (where more than one command is processed in parallel).
_group_end_cmds: list
    A list of message 'commands' that are found at the begining of a 'group'
    (where more than one command is processed in parallel).
__timed_cmds: list.
    A list of message 'commands' that aren't groupable but are timed.
__communication_cmds: list.
    A list of message 'commands' that require network communication.
"""

# GROUP THE DIFFERENT COMMAND TYPES
# The commands that start/end a 'group', but not flyer commands.
_group_start_cmds = ['set', 'trigger', 'kickoff']
_group_end_cmds = ['wait']
# The commands that aren't  groupable but still have statistics.
_timed_cmds = ['stage', 'unstage']
# The commands that only require network communication.
_communication_cmds = ['read', 'describe']
# The commands that emit documents to databroker.
_emit_cmds = ['open_run', 'close_run']
# Define some times associated with network communications ('communication'),
# internal processing ('internal') and databroker document emit time ('emit').
# These are defined in a global dictionary to allow for the status_objects to
# access them.
Process_time = {'communication': _TimeStats(0, 0),
                'internal': _TimeStats(0, 0),
                'emit': _TimeStats(0, 0)}


def est_delta_time(plan):
    '''A generator function that yields an estimate of the time to complete
    each message in 'plan'.

    This method takes in a plan and yields out a named tuple with the structure
    (msg, process_time, action_time). Where msg is the message
    from the plan and process_time/ action_time are (est_time, std_dev) named
    tuples which provide the time delta to process/perform the message
    respectively. Note that for actions that do not return a status object the
    action_time will be 'None' while for 'wait' and 'sleep' messages the
    action_time will also be 'None' and the 'process_time' will be the wait/
    sleep time.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the est_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan,
        and the est_time/ std_dev named tuple for the process and action
        times for the message.
    '''

    # create a reference to a dictionary called plan_history to capture set
    # changes during the plan for later modification.
    plan_history = {'set': {}, 'trigger': {}}
    # create a dictionary for storing the info regarding groups in progress.
    groups = defaultdict(_GroupStats)

    for msg in plan:
        # case 1: commands that are 'groupable'
        if msg.command in _group_start_cmds:
            # add set and trigger commands to plan_history
            if msg.command == 'set':
                plan_history['set'][msg.obj.name] = (msg.args, msg.kwargs)
            elif msg.command == 'trigger':
                if msg.obj.name in list(plan_history['trigger'].keys()):
                    plan_history['trigger'][msg.obj.name] += 1
                else:
                    plan_history['trigger'][msg.obj.name] = 1

            process_time = Process_time['communication']
            action_time = obj_est(msg, plan_history=plan_history)

            try:
                group_time = groups[msg.kwargs.get('group')]
            except (KeyError, TypeError):
                group_time = _GroupStats(_TimeStats(0, 0), _TimeStats(0, 0))

            # update the times for the group.
            new_delta_time = combine_est(group_time.elapsed_time, action_time)
            groups[msg.kwargs.get('group')] = _GroupStats(combine_est(
                                                 group_time.delta_time,
                                                 new_delta_time,
                                                 method='max'),
                                                 group_time.elapsed_time)

            # yield out the times for the message
            yield _MsgStats(msg, process_time, action_time)

        # case 2: commands that end 'groups'
        elif msg.command in _group_end_cmds:
            group_time = groups.pop(msg.kwargs.get('group'))

            process_time = group_time.delta_time
            action_time = None

            yield _MsgStats(msg, process_time, action_time)

        # case 3: commands that have stats but are not 'groupable'
        elif msg.command in _timed_cmds:
            process_time = obj_est(msg, plan_history=plan_history)
            action_time = None

            yield _MsgStats(msg, process_time, action_time)

        # case 4: commands that have an associated communication time
        elif msg.command in _communication_cmds:
            process_time = Process_time['communication']
            action_time = None

            yield _MsgStats(msg, process_time, action_time)

        # case 5: commands that have an associated databroker emit time
        elif msg.command in _communication_cmds:
            process_time = Process_time['emit']
            action_time = None

            yield _MsgStats(msg, process_time, action_time)

        # case 6: sleep command
        elif msg.command is 'sleep':
            process_time = _TimeStats(msg.args[0], 0)
            action_time = None

            yield _MsgStats(msg, process_time, action_time)

        # case 7: commands that only require internal processing
        else:
            process_time = Process_time['internal']
            action_time = None

            yield _MsgStats(msg, process_time, action_time)

        # before moving to a new msg increase all group elapsed times.
        for key in groups:
            if msg.command in _group_end_cmds:
                extra_time = combine_est(group_time.delta_time,
                                         group_time.elapsed_time,
                                         method='subtract')
            else:
                extra_time = process_time

            prev_group = groups[key]
            groups[key] = _GroupStats(prev_group.delta_time,
                                      combine_est(prev_group.elapsed_time,
                                                  extra_time))


def combine_est(est_time_1, est_time_2, method='sum'):
    """Returns the combination of est_time_1 and est_time_2.

    This function returns the combination of the named tuples est_time_1 and
    est_time_2, combined using the method defined by 'method'.

    Parameters
    ----------
    est_time_1, est_time_2 : namedtuples.
        The tuples containing the est_time/std_dev tuples to be combined.
    method : string, optional.
        The method to use for the combination of est_time_1 and est_time_2,
        default is 'sum', other options are 'max', 'min' and 'subtract'.

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
    """Returns the est_time/std_dev named tuple for the object referenced in
    msg.

    This function returns the est_time/std_dev named tuple for the object
    referenced in msg.obj and for the command type referenced in msg.command.

    Parameters
    ----------
    msg : message.
        The message that the estimated time is to be found for.
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
        The combined est_time/std_dev named tuple.
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


def est_time(plan):
    '''A generator function that yields a cumulative estimated time for each
    message in 'plan'.

    This method yields out a named tuple with the structure (message, stop_time
    , start_time). Where message is the message from the plan and stop/start
    times are (est_time, std_dev) named tuples which provides the end/start
    time for each message taking the scan start as zero. 'Wait' messages yield
    the times for the group as a whole.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan and
        the est_time/ std_dev named tuples for the stop and start times for the
        message, 'wait' messages yield times for the group they conclude.
    '''

    # Define the cumulative timeand a group time dictionary for tracking group
    # start times.
    cumulative_time = _TimeStats(0, 0)
    groups = defaultdict(_TimeStats)

    for delta_time in est_delta_time(plan):
        # if it is a groupable command
        if delta_time.message.command in _group_start_cmds:
            # if it is the first command for this group store the start time.
            if delta_time.message.kwargs.get("group") not in groups.keys():
                groups[delta_time.message.kwargs.get("group")] =\
                    cumulative_time

            start_time = cumulative_time
            cumulative_time = combine_est(cumulative_time,
                                          delta_time.process_time)
            stop_time = combine_est(cumulative_time, delta_time.action_time)

            yield _CumulativeStats(delta_time.message, stop_time, start_time)

        # if this is a group ending command.
        elif delta_time.message.command in _group_end_cmds:
            start_time = groups.pop(delta_time.message.kwargs.get('group'))
            stop_time = combine_est(start_time, delta_time.process_time)
            cumulative_time = combine_est(cumulative_time, stop_time,
                                          method='max')

            yield _CumulativeStats(delta_time.message, stop_time, start_time)

        # For all other cases.
        else:
            start_time = cumulative_time
            cumulative_time = combine_est(cumulative_time,
                                          delta_time.process_time)

            yield _CumulativeStats(delta_time.message, cumulative_time,
                                   start_time)


def est_time_per_group(plan):
    '''A generator function that removes any yields from est_time(plan) that
    occur within a group.

    This method yields out a named tuple with the structure (message, stop_time
    , start_time). Where message is the message from the plan and stop/start
    times are (est_time, std_dev) named tuples which provides the end/start
    time for each message taking the scan start as zero. It removes any yields
    from 'groupable' messages but does yield for the group as a whole, at the
    'wait' message that concludes the group.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan and
        the est_time/ std_dev named tuples for the stop and start times for the
        message, 'wait' messages yield times for the group they conclude.
    '''

    for estimated_time in est_time(plan):
        if estimated_time.message.command not in _group_start_cmds:
            yield estimated_time


def est_time_run(plan):
    '''The generator function adds the start/stop times for 'runs' to
    est_time(plan) these will be yielded at the 'close_run' message.

    This method yields out a named tuple with the structure (msg, stop_time
    , start_time. Where msg is the message from the plan and stop/start times
    are (est_time, std_dev) named tuples which provides the end/start time for
    each message taking the scan start as zero. It yields at each message and
    yields a start and stop time for each 'run'for the 'close_run' message.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan and
        the est_time/ std_dev named tuples for the stop and start times for the
        message, 'close_run' messages yield times for the run they conclude.
    '''
    # Note: when parallel runs are allowed this will need updating.
    for estimated_time in est_time(plan):
        if estimated_time.message.command is 'open_run':
            start_time = estimated_time.start_time
            yield estimated_time
        elif estimated_time.message.command is 'close_run':
            stop_time = estimated_time.stop_time
            yield _CumulativeStats(estimated_time.message, stop_time,
                                   start_time)
        else:
            yield estimated_time


def est_time_per_run(plan):
    '''The generator function removes any yields from est_time_run(plan) that
    occur within a run.

    This method yields out a named tuple with the structure (msg, stop_time,
    start_time). Where msg is the message from the plan and stop/start times
    are (est_time, std_dev) named tuples which provides the end/start time for
    each message taking the scan start as zero. It removes any yields from that
    occur between 'open_run' and 'close_run' messages, the combined time for
    the messages between these messages  is returned with the 'close_run'
    message.

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the estimated_time is to be estimated for.

    Returns
    -----------------
    out_time : named tuple.
        A named tuple containing 3 items, the message specifed in the plan and
        the est_time/ std_dev named tuple for stop/start times for the message.
        The times for the runs yield with the 'close_run' message that
        concludes the run.
    '''

    in_run = False
    for estimated_time in est_time_run(plan):
        if estimated_time.message.command == 'open_run':
            in_run = True
            yield estimated_time
        elif estimated_time.message.command == 'close_run':
            in_run = False
            yield estimated_time
        elif not in_run:
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
        time_string = ' -> start:'
        time_string += '{}+/-'.format(estimated_time.start_time.est_time),
        time_string += '{}, stop:'.format(estimated_time.start_time.std_dev),
        time_string += '{}+/-'.format(estimated_time.stop_time.est_time),
        time_string += '{}'.format(estimated_time.stop_time.std_dev)

        if estimated_time.message.command == 'set':
            print('move {}: to '.format(estimated_time.message.obj.name) +
                  '{} '.format(estimated_time.message.args) + time_string)

        elif estimated_time.message.command == 'sleep':
            print('wait for {} '.format(estimated_time.message.args[0]) +
                  time_string)

        elif estimated_time.message.command == 'open_run':
            run_id += 1
            print('open_run {} '.format(run_id) + time_string)

        elif estimated_time.message.command == 'close_run':
            print('close_run {} '.format(run_id) + time_string)

        elif estimated_time.message.command == 'wait':
            print('wait on group ' +
                  '{} '.format(estimated_time.message.kwargs.get('group')) +
                  time_string)

        elif hasattr(estimated_time.message.obj, 'name'):
            print('{} '.format(estimated_time.message.command) +
                  '{} '.format(estimated_time.message.obj.name) + time_string)

        else:
            print('{} '.format(estiamted_time.message.command) + time_string)
