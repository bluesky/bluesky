import sys
from pathlib import Path
from types import TracebackType
from typing import List, Tuple


from IPython import get_ipython


def ipy_plan_exception_sparsifier() -> List[str]:
    """
    plan_traceback_handler that removes some spammy traceback entries.

    Install in your run engine via the plan_traceback_handler kwarg.
    """
    etype, value, tb = sys.exc_info()
    new_tb = remove_matching_frames(tb, ('bluesky.run_engine', 'bluesky.preprocessors'))
    if new_tb is None:
        # Somehow we've deleted all the frames, go back
        new_tb = tb
    ipy = get_ipython()
    return ipy.InteractiveTB.structured_traceback(etype=etype, value=value, tb=new_tb)


def remove_matching_frames(tb: TracebackType, modules: Tuple[str]) -> TracebackType:
    """
    Remove details from a traceback that match a specific module.

    This does not mutate the original traceback. Instead, a
    new traceback object is created that uses the same frames as the
    original inputted traceback. Note that this must be done explicitly
    using the TracebackType constructor because traceback objects
    cannot be copied.

    The tb object typically comes from _, _, tb = sys.exc_info()
    and has the next traceback in the stack stored on its
    tb_next attribute, or None if it was the last traceback.

    Parameters
    ----------
    tb : traceback
        The traceback object as presented to us by the sys module.

    module : str
        The names of the modules to remove from the traceback.
    """
    if tb is None:
        return tb
    if match_frame(tb=tb, modules=modules):
        return remove_matching_frames(
            tb=tb.tb_next,
            modules=modules,
        )
    else:
        return TracebackType(
            tb_next=remove_matching_frames(
                tb=tb.tb_next,
                modules=modules,
            ),
            tb_frame=tb.tb_frame,
            tb_lasti=tb.tb_lasti,
            tb_lineno=tb.tb_lineno,
        )


def match_frame(tb: TracebackType, modules: Tuple[str]) -> bool:
    """
    Returns True if the traceback's frame comes from the given modules.

    Parameters
    ----------
    tb : traceback
        The traceback object as presented to us by the sys module.

    modules : tuple of str
        The names of the modules to remove from the traceback.
    """
    path = Path(tb.tb_frame.f_code.co_filename)
    # Match modules that are just python files
    if path.name in modules:
        return True
    module_parts = [path.stem]
    # Search for the highest path that has an __init__.py file
    while path != path.root:
        if (path.parent / '__init__.py').exists():
            path = path.parent
            module_parts.insert(0, path.name)
        else:
            break
    for module in modules:
        input_parts = module.split('.')
        if input_parts == module_parts[:len(input_parts)]:
            return True
    return False
