from bluesky.examples import Base, Mover, Flyer


def dryrun(gen):
    """
    Replace all hardware objects with mocked objects.
    
    This could catch mistakes that lead to IllegalMessageSequence
    or more basic problems like NameError.

    Note: This does nothing to affect subscriptions, so simulated
    data from the mocked objects will be saved unless the relevant
    callbacks are unsubscribed.
    """
    # TODO Use obj.read_attrs (etc.) to create a better mock.
    for msg in gen:
        live_obj = msg.obj
        if hasattr(live_obj, 'set'):
            mock_obj = Mover
        elif hasattr(live_obj, 'kickoff'):
            mock_obj = Flyer()
        else:
            mock_obj = Base()
        mock_msg = msg._replace(obj=mock_obj)
        yield mock_msg
