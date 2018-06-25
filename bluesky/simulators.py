from warnings import warn
from bluesky.preprocessors import print_summary_wrapper
from collections import namedtuple
import inspect
import sys
import stat

_TimeStats=namedtuple('TimeStats','est_time std_dev')
_MsgStats=namedtuple('MsgStats', 'msg est_time std_dev')


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
    A list of message 'commands' that are found at the begining of a 'group' (where more than
    one command is processed in parallel).
_group_end_cmds: list
    A list of message 'commands' that are found at the begining of a 'group' (where more than
    one command is processed in parallel).
_self._timed_cmds: list.
    A list of message 'commands' that do not start/end a run/group/flyer but are stil timed.
__flyer_start_cmds: list.
    A list of message 'commands' that start a 'flyer'.
_plan_history: dict.
    A dictionary containing information on values updated during the plan. It has key:arg 
    pairs with keys relating to message components where each arg is a dictionary 
    containing devices (detectors, axes) that have been updated, as keywords and a value 
    indicating the update status. For 'set' the update status is the latest 'position' that
    it is to be set and for 'trigger' the update status is the number of times since the 
    beginning of the scan or since an 'unstage' event that the device has been triggered.

"""
#The commands that start/end a 'run'.
_run_start_cmds = ['open_run']
_run_end_cmds = ['close_run']
#The commands that start/end a 'group', but not flyer commands.
_group_start_cmds = ['set','trigger']   
_group_end_cmds = ['wait']
#The commands that don't start/end a 'group/plan' but are still timed.
_timed_cmds = ['stage', 'unstage', 'sleep']
#Commands that are 
_flyer_start_cmds = ['kickoff']
#create a reference to a dictionary called plan_history to capture set changes during the 
#plan for later modification.
_plan_history = {'set':{}, 'trigger':{}}

def est_time(plan, print_output = True):
    '''The call method.

    This method takes in a plan, and an optional print_output kwarg, and returns an estimate for 
    the time to complete for the plan, in addition it returns an time estimate for any subplans. 

    PARAMETERS
    ----------
    plan : generator.
        The bluesky plan that the est_time is to be estimated for.
    print_output : boolean, optional.
        Indicates if the return values should also be printed to the command line in a human 
        readable way, default value is 'True'.

    RETURN PARAMETERS
    -----------------
    out_time : list.
        A list containing 2 items, the est_time and the std_dev, for the plan.
    run_info : list.
        A list of items, 1 for each run in the plan, containing an est_time/std_dev tuple.

    '''
    #reset the ._plan_history dictionary. 
    _plan_history = {'set':{}, 'trigger':{} }

    #Define some variables used in the following.
    out_time = _TimeStats(0, 0) #this holds the plan est_time and std_dev as a pair.
    run_info = [] #used to track the ETA and STD_DEV for any 'runs' inside the plan.

    for msg in plan:
        if msg.command in _run_start_cmds:
            rn_time = run_est(msg, plan)
            run_info.append(rn_time)
            out_time = combine_est(out_time, rn_time)
        if msg.command in (_group_start_cmds + _flyer_start_cmds):
            grp_time = group_est(msg, plan)
            out_time = combine_est(out_time, grp_time)            

        else:
            object_est = obj_est(msg)
            out_time = combine_est(out_time, object_est)

        if print_output == True:
            for i, run in enumerate(run_info):
                print('  * Run %d est. time --> %.2g s, std Dev --> %.2g s' % (i+1, run[0], run[1]))
            print('Plan est. time --> %.2g s, std Dev --> %.2g s' % (out_time.est_time, 
                                                                                out_time.std_dev))

        return out_time, run_info

def msg_est(plan):
    '''Returns a list of msg/est_time/std_dev named tuples for each msg in plan

    This function is used to return an un-grouped list of msgs and estimated times. It differs 
    from self.est_time in that it does not estimate the entire plan, or 'runs' in the plan and it 
    does not sort into msg 'groups', which are done in parallel. it simply returns a msg/est_time/
    std_dev tuple for each message in the plan.

    PARAMETERS
    ----------
    plan : generator.
        The bluesky plan that the est_time is to be estimated for.

    RETURN PARAMETERS
    -----------------
    out_time : list.
        A list containing a msg/est_time/std_dev named tuple for each message in the plan.
    '''
    
    out_time = []
    for msg in plan:
        est_time = obj_est(msg)
        out_time.append(_MsgStats( msg, est_time.est_time, est_time.std_dev))

    return out_time

         

def combine_est(est_time_1, est_time_2, method = 'sum'):
    """
    Returns the combination est_time/std_dev pairs et_1 and et_2.
    This function returns the combination of est_time_1 and est_time_2, combined using the 
    method definedby 'method'.

    Parameters
    ----------
    est_time_1, est_time_2 : namedtuples.
        The tuples containing the est_time/std_dev tuples to be combined.
    method : string, optional.
        The method to use for the combination of est_time_1 and est_time_2, default is 
        'sum'.

    Return Parameters
    -----------------
    out_time : namedtuple.
        The combined est_time/std_dev tuple.
    """
    
    if method == 'sum':
        #adds the est_time, uses quadrature for std_dev.
        out_time = _TimeStats(est_time_1.est_time + est_time_2.est_time, 
                        (est_time_1.std_dev**2 + est_time_2.std_dev**2)**.5)

    if method == 'subtract':
        #adds the est_time, uses quadrature for std_dev.
        out_time = _TimeStats(est_time_1.est_time - est_time_2.est_time, 
                        (est_time_1.std_dev**2 + est_time_2.std_dev**2)**.5)

    elif method == 'max':
        #finds the max est_time, finds the potentially largest difference from the max_time
        #as the std_dev.
        max_time = max(est_time1.est_time, est_time2.est_time) 
        pos_stdev = max(est_time1.est_time + est_time1.std_dev,
                        est_time2.est_time + est_time2.std_dev )
        neg_stdev = min(est_time1.est_time - est_time1.std_dev,
                        est_time2.est_time - est_time2.std_dev )
        max_stdev = max(abs(pos_stdev-max_time), abs(neg_stdev-max_time))
        out_time = _TimeStats(max_time, max_stdev)

    return out_time



def obj_est(msg ):
    """
    Returns the est_time/std_dev pair for the object referenced in msg.
    This function returns the est_time/std_dev pair for the object referenced in msg.obj and 
    for the command type reference in msg.command.

    Parameters
    ----------
    msg : message.
        The first message of the group that was detected in plan_ETA.
    
    Return Parameters
    -----------------
    out_time : list.
        The combined est_time/std_dev pair.

    Updated Parameters
    ------------------
    _plan_history : dict.
        A dictionary containing information on values updated during the plan. It has key:arg 
        pairs with keys relating to message components where each arg is a dictionary 
        containing devices (detectors, axes) that have been updated, as keywords and a value 
        indicating the update status. For 'set' the update status is the latest 'position' that
        it is to be set and for 'trigger' the update status is the number of times since the 
        beginning of the scan or since an 'unstage' event that the device has been triggered.
        NOTE: Any 'set' ot 'trigger' messages processed in this function will be added to the 
        dictionary
    """

    set_dict={}

    if msg.obj is not None:

        if msg.command is 'sleep':
            return [msg.args[0], 0]

        elif msg.command in (_run_start_cmds + _group_start_cmds + _timed_cmds):
            obj =  msg.obj
            #The if-elif section below is used to track how many 'triggers'/'sets' have 
            #occured since the last 'unstage'/start of plan.
            if msg.command == 'unstage':
                _plan_history['trigger'][msg.obj.name] = 0
            elif msg.command == 'trigger':
                if msg.obj.name in list(_plan_history['trigger'].keys()):
                    _plan_history['trigger'][msg.obj.name] += 1
                else:
                    _plan_history['trigger'][msg.obj.name] = 1
            elif msg.command == 'set':
                try:
                    set_dict['start_pos'] = _plan_history['set'][msg.obj.name]
                except KeyError:
                    set_dict['start_pos'] = msg.obj.position

                set_dict['target'] = msg.args[0]
                _plan_history['set'][msg.obj.name] = msg.args[0]

            #determine what values are required for time estimation and find them
            obj_est_time = getattr(msg.obj.est_time, msg.command)
            arg_names = inspect.signature(obj_est_time).parameters
            args = []
            for arg_name in arg_names:
                try:
                    args.append(set_dict[arg_name])
                except KeyError:
                    try: 
                        args.append(_plan_history['set'][arg_name])
                    except KeyError:
                        try:
                            args.append(getattr(obj, arg_name).position)
                        except AttributeError:
                            print('{}.est_time.{} requires a {} attribute on {} but none can be \
                                found'.format(obj.name, msg.command, arg_name, obj) )
                            raise

            #ask the object for a time estimation. 
            object_est = obj_est_time(*args)

            return object_est

        elif msg.command == _flyer_start_cmds:
            #This section pulls out the list of motor positions from the flyer
            #and pulls out the ETA for each step.
            obj = msg.obj
            out_time = (0,0)
            for pos in msg.obj._steps:
                 #determine what values are required for time estimation and find them
                obj_est_time = getattr(obj.est_time,msg.cmd)
                arg_names = inspect.signature(obj_est_time)
                args = []
                for arg_name in arg_names:
                    try: 
                        args.append(_plan_history['set'][arg_name])
                    except KeyError:
                        try:
                            args.append(getattr(obj, arg_name).position)
                        except AttributeError:
                            print('{}.est_time.{} requires a {} attribute but none can be \
                                found'.format(obj.name, msg.command, arg_name) )
                        raise

                #ask the object for a time estimation. 
                object_est = obj_est_time(*args)

                out_time = combine_est(out_time, object_est)
                _plan_history['set'][msg.obj._mot.name] = pos

            return out_time              

        else:
            return _TimeStats(0, 0)

    else:
        return _TimeStats(0, 0)


def group_est(msg, plan):
    """
    Returns the est_time/std_dev tuple for a group.
    This function returns an est_time/std_dev tuple, for a group, where a group is defined as a 
    series of messages which are run simultaneously and is ended on a wait messages. It 
    assumes that it is called from inside plan_est_time, once the first message of the group 
    has been detected. 

    Parameters
    ----------
    msg : message.
        The first message of the group that was detected in plan_ETA.
    plan: generator.
        The generator that has the list of msgs to be examined.
    
    Return Parameters
    -----------------
    out_time : list.
        The combined est_time/std_dev pair.

    Updated Parameters
    ------------------
    _plan_history : dict.
        A dictionary containing information on values updated during the plan. It has key:arg 
        pairs with keys relating to message components where each arg is a dictionary 
        containing devices (detectors, axes) that have been updated, as keywords and a value 
        indicating the update status. For 'set' the update status is the latest 'position' that
        it is to be set and for 'trigger' the update status is the number of times since the 
        beginning of the scan or since an 'unstage' event that the device has been triggered.
        NOTE: Any 'set' ot 'trigger' messages processed in this function will be added to the 
        dictionary

    """
    group_time = {}
    group_time[msg.group] = [obj_est(msg, plan_history = _plan_history)]

    while group_time:           
        #the below if-elif statement is used to track any changes of values using 'set', and
        #the number of 'triggers' called for later use in the time estimate.
        if msg.command == 'set': 
            _plan_history['set'][msg.obj.name] = msg.args[0]

        elif msg.command == 'trigger': 
            if msg.obj.name in list(_plan_history['trigger'].keys() ):
                _plan_history['trigger'][msg.obj.name] += 1

            else:
                _plan_history['trigger'][msg.obj.name] = 1

        #this section estimates the time fro each command.
        msg = next(plan)
        
        if msg.command in self._group_end_cmds:
            group_list = group_time.pop(msg.group)
            out_time = group_list.pop()
            while group_list:
                out_time = combine_est(out_time, group_list.pop(), method = 'max')
    
            for group_name in group_time:    
                for i, item in enumerate(group_time[group_name]):
                    group_time[group_name][i] = combine_est(item, out_time, method = 'subtract')
        else:
            try:
                group_time[msg.group].append(obj_est(msg, plan_history = _plan_history))
            except KeyError:
                group_time[msg.group] = [obj_est(msg, plan_history = _plan_history)]
        
    return out_time


def run_est(msg, plan):
    """
    Returns the est_time/std_dev pair for a run.
    This function returns an est_time/std_dev pair, for a group, where a group is defined as a 
    series of messages which are run simultaneously and is ended on a wait messages. It 
    assumes that it is called from inside plan_est_time, once an 'open_run' has been detected. 

    Parameters
    ----------
    msg : message.
        The first message of the group that was detected in plan_est_time.
    plan : generator.
        The generator that contains the list of messages to be examined.
    
    Return Parameters
    -----------------
    out_time : list.
        The combined est_time/std_dev pair.

    Updated Parameters
    ------------------
    _plan_history : dict.
        A dictionary containing information on values updated during the plan. It has key:arg 
        pairs with keys relating to message components where each arg is a dictionary 
        containing devices (detectors, axes) that have been updated, as keywords and a value 
        indicating the update status. For 'set' the update status is the latest 'position' that
        it is to be set and for 'trigger' the update status is the number of times since the 
        beginning of the scan or since an 'unstage' event that the device has been triggered.
        NOTE: Any 'set' ot 'trigger' messages processed in this function will be added to the 
        dictionary

    """

    out_time = _TimeStats(0,0)
    
    while msg.command not in self._run_end_cmds:
        msg = next(plan)
        if msg.command in (self._run_start_cmds + self._flyer_start_cmds):
            grp_time = self.group_est(msg, plan)
            out_time = self.combine_est(out_time, grp_time)

        else:
            object_est = obj_est(msg)
            out_time = combine_est(out_time, object_est)
    return out_time
