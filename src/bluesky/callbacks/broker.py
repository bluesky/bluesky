import time as ttime

from ..utils import ensure_uid

cursor_up = "\x1b[1a"
check_mark = "\u2713"


def print_checkmark_on_previous_line():
    print(f"{cursor_up}{check_mark}")


def post_run(callback, db, fill=False):
    """Trigger a callback to process all the Documents from a run at the end.

    This function does not receive the Document stream during collection.
    It retrieves the complete set of Documents from the DataBroker after
    collection is complete.

    Parameters
    ----------
    callback : callable
        Expected signature ::

            def func(doc_name, doc):
                pass

    db : Broker
        The databroker instance to use

    fill : boolean, optional
        Whether to deference externally-stored data in the documents.
        False by default.

    Returns
    -------
    func : function
        a function that accepts a RunStop Document

    Examples
    --------
    Print a table with full (lossless) result set at the end of a run.

    >>> table = LiveTable(['det', 'motor'])
    >>> RE(scan(motor, [det], [1,2,3]), {'stop': post_run(table)})
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |           det  |         motor  |
    +------------+-------------------+----------------+----------------+
    |         3  |  14:02:32.218348  |          5.00  |          3.00  |
    |         2  |  14:02:32.158503  |          5.00  |          2.00  |
    |         1  |  14:02:32.099807  |          5.00  |          1.00  |
    +------------+-------------------+----------------+----------------+

    """

    def f(name, doc):
        if name != "stop":
            return
        uid = ensure_uid(doc["run_start"])
        header = db[uid]
        for name, doc in header.documents(fill=fill):
            callback(name, doc)
        # Depending on the order that this callback and the
        # databroker-insertion callback were called in, the databroker might
        # not yet have the 'stop' document that we currently have, so we'll
        # use our copy instead of expecting the header to include one.
        if name != "stop":
            callback("stop", doc)

    return f


def make_restreamer(callback, db):
    """
    Run a callback whenever a uid is updated.

    Parameters
    ----------
    callback : callable
        expected signature is `f(name, doc)`

    db : Broker
        The databroker instance to use

    Example
    -------
    Run a callback whenever a uid is updated.

    >>> def f(name, doc):
    ...     # do stuff
    ...
    >>> g = make_restreamer(f, db)

    To use this `ophyd.callbacks.LastUidSignal`:

    >>> last_uid_signal.subscribe(g)
    """

    def cb(value, **kwargs):
        return db.process(db[value], callback)

    return cb


def verify_files_saved(name, doc, db):
    "This is a brute-force approach. We retrieve all the data."
    ttime.sleep(0.1)  # Wait for data to be saved.
    if name != "stop":
        return
    print("  Verifying that all the run's Documents were saved...")
    try:
        header = db[ensure_uid(doc["run_start"])]
    except Exception as e:
        print(f"  Verification Failed! Error: {e}")
        return
    else:
        print_checkmark_on_previous_line()
    print("  Verifying that all externally-stored files are accessible...")
    try:
        list(db.get_events(header, fill=True))
    except Exception as e:
        print(f"  Verification Failed! Error: {e}")
    else:
        print_checkmark_on_previous_line()
