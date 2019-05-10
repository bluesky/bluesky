def guess_dimensions(start_doc):
    """
    Parameters
    ----------
    Prepare a guess about the dimensions (independent variables).
    start_doc : dict

    Returns
    -------
    dimensions : list
        looks like a plan's 'dimensions' hint, but guessed from heuristics
    """
    motors = start_doc.get('motors')
    if motors is not None:
        return [([motor], 'primary') for motor in motors]
        # For example, if this was a 2D grid scan, we would have:
        # [(['x'], 'primary'), (['y'], 'primary')]
    else:
        # There is no motor, so we will guess this is a time series.
        return [(['time'], 'primary')]


def extract_hints_info(start_doc):
    """
    Parameters
    ----------
    start_doc : dict

    Returns
    -------
    stream_name, dim_fields, all_dim_fields
    """
    plan_hints = start_doc.get('hints', {})
    cleanup_motor_heuristic = False

    # Use the guess if there is not hint about dimensions.
    dimensions = plan_hints.get('dimensions')
    if dimensions is None:
        cleanup_motor_heuristic = True
        dimensions = guess_dimensions(start_doc)

    # Do all the 'dimensions' belong to the same Event stream? If not, that is
    # too complicated for this implementation, so we ignore the plan's
    # dimensions hint and fall back on guessing.
    if len(set(stream_name for fields, stream_name in dimensions)) != 1:
        cleanup_motor_heuristic = True
        dimensions =  guess_dimensions(start_doc)
        warn("We are ignoring the dimensions hinted because we cannot "
             "combine streams.")

    # for each dimension, choose one field only
    # the plan can supply a list of fields. It's assumed the first
    # of the list is always the one plotted against
    # fields could be just one field, like ['time'] or ['x'], but for an "inner
    # product scan", it could be multiple fields like ['x', 'y'] being scanned
    # over jointly. In that case, we just plot against the first one.
    dim_fields = [first_field for (first_field, *_), stream_name in dimensions]

    # Make distinction between flattened fields and plotted fields.
    # Motivation for this is that when plotting, we find dependent variable
    # by finding elements that are not independent variables
    all_dim_fields = [field
                      for fields, stream_name in dimensions
                           for field in fields]

    # Above we checked that all the dimensions belonged to the same Event
    # stream, so we can take the stream_name from any item in the list of
    # dimensions, and we'll get the same result. Might as well use the first
    # one.
    _, dim_stream = dimensions[0]  # so dim_stream is like 'primary'
    # TO DO -- Do we want to return all of these? Maybe we should just return
    # 'dimensions' and let the various callback_factories do whatever
    # transformations they need.
    return dim_stream, dim_fields, all_dim_fields


def hinted_fields(descriptor):
    # Figure out which columns to put in the table.
    obj_names = list(descriptor['object_keys'])
    # We will see if these objects hint at whether
    # a subset of their data keys ('fields') are interesting. If they
    # did, we'll use those. If these didn't, we know that the RunEngine
    # *always* records their complete list of fields, so we can use
    # them all unselectively.
    columns = []
    for obj_name in obj_names:
        try:
            fields = descriptor.get('hints', {}).get(obj_name, {})['fields']
        except KeyError:
            fields = descriptor['object_keys'][obj_name]
        columns.extend(fields)
    return columns
