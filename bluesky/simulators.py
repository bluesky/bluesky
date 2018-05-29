from warnings import warn
from bluesky.preprocessors import print_summary_wrapper


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


def est_time(plan, print_output = True):
    """
    Estimates a time for a plan to be completed.
    This function estimates the time it will take (est_time) to complete the plan defined by 
    'plan'. This is done by generating an est_time for each of the steps in the plan using the 
    est_time attribute on the devices used in each step. As the device est_time is generally based 
    on statistics gathered from previous use, and returns the mean value (est_time) and the 
    standard deviation (std_dev) from those statistics, est_time also returns the std_dev for the 
    est_time to give an idea of the accuracy of the prediction. As this is a statistical approach 
    to time estimating at the device level it is expected that the accuracy will improve with more 
    use of the device through bluesky. The final information returned is a list of est_time, std_dev
    pairs for each 'run' in the plan, where a 'run' is defined as the anything that occurs between 
    a 'start' and 'stop' document being generated (which is how the data is stored in the 
    databroker). 

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the est_time is to be estimated for.
    print_output : boolean, optional.
        Indicates if the return values should also be printed to the command line in a human 
        readable way, default value is 'True'.
    
    Return Parameters
    -----------------
    out_est_time : list.
        A list containing 2 items, the est_time and the std_dev, for the plan.
    run_info : list.
        A list of items, 1 for each run in the plan, containing an est_time/std_dev tuple.

    """

    def combine_est_time(est_time_1, est_time_2, method = 'sum'):
        """
        Returns the combination est_time/std_dev pairs et_1 and et_2.
        This function returns the combination of est_time_1 and est_time_2, combined using the 
        method definedby 'method'.

        Parameters
        ----------
        est_time_1, est_time_2 : tuples.
            The tuples containing the est_time/std_dev tuples to be combined.
        method : string, optional.
            The method to use for the combination of est_time_1 and est_time_2, default is 
            'sum'.

        Return Parameters
        -----------------
        out_est_time : list.
            The combined est_time/std_dev tuple.
        """
        out_est_time = [0,0]
        print (f'est_time_1 = {est_time_1}: est_time_2 = {est_time_2}') 
        if method == 'sum':
            out_est_time[0] = est_time_1[0] + est_time_2[0]
            out_est_time[1] = est_time_1[1] + est_time_2[1]

        elif method == 'max':
            if est_time_1[0] < est_time_2[0]:
                out_est_time = est_time_2 

            else:
                out_est_time = est_time_1

        return out_est_time



    def obj_est_time(msg, val_dict):
        """
        Returns the est_time/std_dev pair for the object referenced in msg.
        This function returns the est_time/std_dev pair for the object referenced in msg.obj and 
        for the command type reference in msg.command.

        Parameters
        ----------
        msg : message.
            The first message of the group that was detected in plan_ETA.
        val_dict : dict.
            A dictionary containing information on values updated during the plan. It has key:arg 
            pairs with keys relating to message components where each arg is a dictionary 
            containing devices (detectors, axes) that have been updated, as keywords and a value 
            indicating the update status. For 'set' the update status is the latest 'position' that
            it is to be set and for 'trigger' the update status is the number of times since the 
            beginning of the scan or since an 'unstage' event that the device has been triggered.
        
        Return Parameters
        -----------------
        out_est_time : list.
            The combined est_time/std_dev pair.
        val_dict : dict.
            The updated version of val_dict.
        """

        if msg.obj is not None:

            if msg.command in ['set', 'trigger', 'stage', 'unstage']:
                obj =  msg.obj
                if msg.command == 'unstage':
                    val_dict['trigger'][msg.obj.name] = 0
                elif msg.command == 'trigger':
                    if msg.obj.name in list(val_dict['trigger'].keys()):
                        val_dict['trigger'][msg.obj.name] += 1
                    else:
                        val_dict['trigger'][msg.obj.name] = 1
                object_est_time, val_dict = obj.est_time(cmd = msg.command, val_dict = val_dict, 
                                               vals = msg.args)
                return object_est_time, val_dict

            elif msg.command == 'kickoff':
                #and adds the ETA for each step. 
                #This section pulls out the list of motor positions from the flyer
                obj = msg.obj
                out_est_time = (0,0)
                for pos in msg.obj._steps:
                    object_est_time, val_dict = obj.est_time(cmd = 'set', val_dict = val_dict, 
                                                            vals = [pos])
                    out_est_time = combine_est_time(out_est_time, object_est_time)
                    val_dict['set'][msg.obj._mot.name] = pos

                return out_est_time, val_dict                
    
            else:
                return [0, 0], val_dict

        elif msg.command is 'sleep':
            return [msg.args[0], 0], val_dict

        else:
            return [0, 0], val_dict


    def group_est_time(msg, val_dict):
        """
        Returns the est_time/std_dev tuple for a group.
        This function returns an est_time/std_dev tuple, for a group, where a group is defined as a 
        series of messages which are run simultaneously and is ended on a wait messages. It 
        assumes that it is called from inside plan_est_time, once the first message of the group 
        has been detected. The routine also returns an updated version of val_dict including any 
        changed values from the group.

        Parameters
        ----------
        msg : message.
            The first message of the group that was detected in plan_ETA.
        val_dict : dict.
            A dictionary containing information on values updated during the plan. It has key:arg 
            pairs with keys relating to message components where each arg is a dictionary 
            containing devices, (detectors, axes) that have been updated, as keywords and a value 
            indicating the update status. For 'set' the update status is the latest 'position' 
            that it has reached and for 'trigger' the update status is the number of times since 
            the beginning of the scan or since an 'unstage' event that the device has been 
            triggered.
        
        Return Parameters
        -----------------
        out_est_time : list.
            The combined est_time/std_dev pair.
        val_dict : dict.
            The updated version of val_dict.
        """
        out_est_time, val_dict = obj_est_time(msg, val_dict)

        while msg.command is not 'wait':
            if msg.command == 'set': 
                val_dict['set'][msg.obj.name] = msg.args[0]

            elif msg.command == 'trigger': 
                if msg.obj.name in list( val_dict['trigger'].keys() ):
                    val_dict['trigger'][msg.obj.name] += 1

                else:
                    val_dict['trigger'][msg.obj.name] = 1

            msg = next(plan)
            object_est_time, val_dict = obj_est_time(msg, val_dict)
            out_est_time = combine_est_time(out_est_time, object_est_time, method = 'max')

        return out_est_time, val_dict


    def run_est_time(msg, val_dict):
        """
        Returns the est_time/std_dev pair for a run.
        This function returns an est_time/std_dev pair, for a group, where a group is defined as a 
        series of messages which are run simultaneously and is ended on a wait messages. It 
        assumes that it is called from inside plan_est_time, once an 'open_run' has been detected. 
        The routine also returns an updated version of val_dict including any changed values 
        from the run.

        Parameters
        ----------
        msg : message.
            The first message of the group that was detected in plan_est_time.
        val_dict : dict.
            A dictionary containing information on values updated during the plan. It has key:arg 
            pairs with keys relating to message components where each arg is a dictionary 
            containing devices, (detectors, axes) that have been updated, as keywords and a value 
            indicating the update status. For 'set' the update status is the latest 'position' 
            that it has reached and for 'trigger' the update status is the number of times since 
            the beginning of the scan or since an 'unstage' event that the device has been 
            triggered.
        
        Return Parameters
        -----------------
        out_est_time : list.
            The combined est_time/std_dev pair.
        val_dict : dict.
            The updated version of val_dict.
        """

        out_est_time = [0,0]
        
        while msg.command is not 'close_run':
            msg = next(plan)
            if msg.command in ['set','trigger','kickoff']:
                grp_est_time, val_dict = group_est_time(msg, val_dict)
                out_est_time = combine_est_time(out_est_time, grp_est_time)

            else:
                object_est_time, val_dict = obj_est_time(msg, val_dict)
                out_est_time = combine_est_time(out_est_time, object_est_time)
   
        return out_est_time, val_dict


    #Define some variables used in the following.
    val_dict = {'set':{}, 'trigger':{} } #this holds information on the updated values.
    out_est_time = [0, 0] #this holds the plan est_time and std_dev as a pair.
    run_info = [] #used to track the ETA and STD_DEV for any 'runs' inside the plan.

    for msg in plan:
        if msg.command is 'open_run':
            rn_est_time, val_dict = run_est_time(msg, val_dict)
            run_info.append(rn_est_time)
            out_est_time = combine_est_time(out_est_time, rn_est_time)
        if msg.command in ['set','trigger','kickoff']:
            grp_est_time, val_dict = group_est_time(msg, val_dict)
            out_est_time = combine_est_time(out_est_time, grp_est_time)            

        else:
            object_est_time, val_dict = obj_est_time(msg, val_dict)
            out_est_time = combine_est_time(out_est_time, object_est_time)

    if print_output == True:
        for i, run in enumerate(run_info):
            print('  * Run %d est. time --> %.2f s, std Dev --> %.2f s' % (i+1, run[0], run[1]))
        print('Plan est. time --> %.2f s, std Dev --> %.2f s' % (out_est_time[0], out_est_time[1]))

    return out_est_time, run_info



