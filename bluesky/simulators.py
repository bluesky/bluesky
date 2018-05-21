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


def plan_ETA(plan, print_output = True):
    """
    Estimates a time for a plan to be completed.
    This function estimates the time it will take (ETA) to complete the plan defined by 'plan'. This    is done by generating an ETA for each of the steps in the plan using the ETA attribute on the 
    devices used in each step. As the device ETA is generally based on statistics gathered from 
    previous use, and returns the mean value (ETA) and the standard deviation (STD_DEV) from those 
    statistics, plan_ETA also returns the STD_DEV for the ETA to give an idea of the accuracy of 
    the prediction. As this is a statistical approach to time estimating at the device level it is
    expected that the accuracy will improve with more use of the device through bluesky. The final
    information returned is a list of ETA, STD_DEV pairs for each 'run' in the plan, where a 'run' 
    is defined as the anything that occurs between a 'start' and 'stop' document being generated 
    (which is how the data is stored in the databroker). 

    Parameters
    ----------
    plan : generator.
        The bluesky plan that the ETA is to be estimated for.
    print_output : boolean, optional.
        Indicates if the return values should also be printed to the command line in a human 
        readable way, default value is 'True'.
    
    Return Parameters
    -----------------
    out_ETA : list.
        A list containing 2 items, the ETA and the STD_DEV, for the plan.
    run_info : list.
        A list of items, 1 for each run in the plan, containing the ETA and STD_DEV.

    """

    def combine_ETA(ETA_1, ETA_2, method = 'sum'):
        """
        Returns the combination ETA/STD_DEV pairs ETA_1 and ETA_2.
        This function returns the combination of ETA_1 and ETA_2, combined using the method defined
        by 'method'.

        Parameters
        ----------
        ETA_1, ETA_2 : list.
            The lists containing the ETA/STD_DEV pairs to be combined.
        method : string, optional.
            The method to use for the combination of ETA_1 and ETA_2, default is 'sum'.

        Return Parameters
        -----------------
        out_ETA : list.
            The combined ETA/STD_DEV pair.
        """
        out_ETA = [0,0]
        if method == 'sum':
            out_ETA[0] = ETA_1[0] + ETA_2[0]
            out_ETA[1] = ETA_1[1] + ETA_2[1]

        elif method == 'max':
            if ETA_1[0] < ETA_2[0]:
                out_ETA = ETA_2 

            else:
                out_ETA = ETA_1

        return out_ETA



    def obj_ETA(msg, val_dict):
        """
        Returns the ETA/STD_DEV pair for the object referenced in msg.
        This function returns the ETA/STD_DEV pair for the object referenced in msg.obj and for the 
        command type reference in msg.command.

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
        out_ETA : list.
            The combined ETA/STD_DEV pair.
        val_dict : dict.
            The updated version of val_dict.
        """

        if msg.obj is not None:

            if msg.command in ['set', 'trigger', 'stage', 'unstage']:
                obj = getattr(sys.modules[__name__], msg.obj.name)
                if msg.command == 'unstage':
                    val_dict['trigger'][msg.obj.name] = 0
                object_ETA = obj.ETA(cmd = msg.command, val_dict = val_dict, vals = msg.args)
                return object_ETA, val_dict

            elif msg.command == 'kickoff':
                #and adds the ETA for each step. 
                #This section pulls out the list of motor positions from the flyer
                obj = getattr(sys.modules[__name__], msg.obj._mot.name)
                out_ETA = [0,0]
                for pos in msg.obj._steps:
                    object_ETA = obj.ETA(cmd = 'set', val_dict = val_dict, vals = [pos])
                    out_ETA = combine_ETA(out_ETA, object_ETA)
                    val_dict['set'][msg.obj._mot.name] = pos

                return out_ETA, val_dict                
    
            else:
                return [0, 0], val_dict

        elif msg.command is 'sleep':
            return [msg.args[0], 0], val_dict

        else:
            return[0, 0], val_dict


    def group_ETA(msg, val_dict):
        """
        Returns the ETA/STD_DEV pair for a group.
        This function returns an ETA/STD_DEV pair, for a group, where a group is defined as a 
        series of messages which are run simultaneously and is ended on a wait messages. It 
        assumes that it is called from inside plan_ETA, once the first message of the group has 
        been detected. The routine also returns an updated version of val_dict including any 
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
        out_ETA : list.
            The combined ETA/STD_DEV pair.
        val_dict : dict.
            The updated version of val_dict.
        """
        out_ETA, val_dict = obj_ETA(msg, val_dict)

        while msg.command is not 'wait':
            if msg.command == 'set': 
                val_dict['set'][msg.obj.name] = msg.args[0]

            elif msg.command == 'trigger': 
                if msg.obj.name in list( val_dict['trigger'].keys() ):
                    val_dict['trigger'][msg.obj.name] += 1

                else:
                    val_dict['trigger'][msg.obj.name] = 1

            msg = next(plan)
            object_ETA, val_dict = obj_ETA(msg, val_dict)
            out_ETA = combine_ETA(out_ETA, object_ETA, method = 'max')
            if out_ETA[0] < object_ETA[0]:
                out_ETA = object_ETA

        return out_ETA, val_dict


     def run_ETA(msg, val_dict):
        """
        Returns the ETA/STD_DEV pair for a run.
        This function returns an ETA/STD_DEV pair, for a group, where a group is defined as a 
        series of messages which are run simultaneously and is ended on a wait messages. It 
        assumes that it is called from inside plan_ETA, once an 'open_run' has been detected. 
        The routine also returns an updated version of val_dict including any changed values 
        from the run.

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
        out_ETA : list.
            The combined ETA/STD_DEV pair.
        val_dict : dict.
            The updated version of val_dict.
        """

        out_ETA = [0,0]
        
        while msg.command is not 'close_run':
            msg = next(plan)
            if msg.command in ['set','trigger','kickoff']:
                grp_ETA, val_dict = group_ETA(msg, val_dict)
                out_ETA = combine_ETA(out_ETA, grp_ETA)

            else:
                object_ETA, val_dict = obj_ETA(msg, val_dict)
                out_ETA = combine_ETA(out_ETA, object_ETA)
   
        return out_ETA, val_dict


    #Define some variables used in the following.
    val_dict = {'set':{}, 'trigger':{} } #this holds information on the updated values.
    out_ETA = [0, 0] #this holds the plan ETA and STD_DEV as a pair.
    run_info = [] #used to track the ETA and STD_DEV for any 'runs' inside the plan.

    for msg in plan:
        if msg.command is 'open_run':
            rn_ETA, val_dict = run_ETA(msg, val_dict)
            run_info.append(rn_ETA)
            out_ETA = combine_ETA(out_ETA, rn_ETA)
        if msg.command in ['set','trigger','kickoff']:
            grp_ETA, val_dict = group_ETA(msg, val_dict)
            out_ETA = combine_ETA(out_ETA, grp_ETA)            

        else:
            object_ETA, val_dict = obj_ETA(msg, val_dict)
            out_ETA = combine_ETA(out_ETA, object_ETA)

    if print_output == True:
        for i, run in enumerate(run_info):
            print('  * Run %d ETA --> %.2f s, Std Dev --> %.2f s' % (i+1, run[0], run[1]))
        print('Plan ETA --> %.2f s, Std Dev --> %.2f s' % (out_ETA[0], out_ETA[1]))

    return out_ETA, run_info



