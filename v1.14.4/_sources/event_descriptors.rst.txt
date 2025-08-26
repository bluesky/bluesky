Event Descriptors
=================

In the section on :doc:`documents`, we gave an overview of the four kinds of
document. We presented an example Run Start, Event, and Run Stop, but we
deferred detailed discussion of the Event Descriptor.

Recall our example 'event' document.

.. code-block:: python

    # 'event' document (same as above, shown again for reference)
    {'data':
        {'temperature': 5.0,
          'x_setpoint': 3.0,
          'x_readback': 3.05},
     'timestamps':
        {'temperature': 1442521007.9258342,
         'x_setpoint': 1442521007.5029348,
         'x_readback': 1442521007.5029348},
     'time': 1442521007.3438923,
     'seq_num': 1,
     'uid': '<randomly-generated unique ID>', 
     'descriptor': '<reference to a descriptor document>'}

Typically, an experiment generates multiple event documents with the same data
keys. For example, there might be ten sequential readings, generating ten event
documents like the one above --- with different readings and timestamps but
identical data keys. All these events refer back to a 'descriptor' with
metadata about the data keys and the configuration of the devices involved.

.. note:: 

    We got the term "data keys" from ``event['data'].keys()``. Again, in our
    example, the data keys are ``['temperature', 'x_setpoint', 'x_readback']``

Data Keys
---------

First, the descriptor provides metadata about each data key.

* dtype --- 'number', 'string', 'array', or 'object' (dict)
* shape --- ``None`` or a list of dimensions like ``[5, 5]`` for a 5x5 array
* source --- a description of the hardware that uniquely identifies it, such as
  an EPICS Process Variable
* (optional) external --- a string specifying where external data, such as a
  large image array, is stored

Arbitrary additional fields are allowed, such as precision or units.
The RunEngine obtains this information from each device it sees by calling
``device.describe()``.

.. code-block:: python

    # excerpt of a 'descriptor' document
    {'data_keys':
        {'temperature':
            {'dtype': 'number',
             'source': '<descriptive string>',
             'shape': [],
             'units': 'K',
             'precision': 3},
         'x_setpoint':
            {'dtype': 'number',
             'source': '<descriptive string>',
             'shape': [],
             'units': 'mm',
             'precision': 2},
         'x_readback':
            {'dtype': 'number',
             'source': '<descriptive string>',
             'shape': [],
             'units': 'mm',
             'precision': 2}},
     ...}

Object Keys
-----------

The ``object_keys`` provide an association between each device and its data keys.

This is needed because a given device can produce multiple data keys. For
example, suppose the ``x_readback`` and ``x_setpoint`` data keys in our example
came from the same device, a motor named ``'x'``.

.. code-block:: python

    # excerpt of a 'descriptor' document
    {'object_keys':
        {'x': ['x_setpoint', 'x_readback'],
         'temp_ctrl': ['temperature']},
     ...}

Specifically, it maps ``device.name`` to ``list(device.describe())``.

Configuration
-------------

Complex devices often have many parameters that do not need to be read anew
with every data point. They are "configuration," by which we mean they don't
typically change in the middle of a run. A detector's exposure time is usually
(but not always) in this category.

Devices delineate between the two by providing two different methods that the
RunEngine can call: ``device.read()`` returns normals readings that are *not*
considered configuration; ``device.read_configuration()`` returns the readings
that are considered configuration.

The first time during a run that the RunEngine is told to read a device, it
reads the device's configuration also. The return value of
``device.describe_configuration()`` is recorded in
``configuration[device.name]['data_keys']``. The return value of
``device.read_configuration()`` is collated into
``configuration[device.name]['data']`` and
``configuration[device.name]['timestamps']``.

In this example, ``x`` has one configuration data key, and ``temp_ctrl``
happens to provide no configuration information.

.. code-block:: python

    # excerpt of a 'descriptor' document
    {'configuration':
        {'x':
           {'data': {'offset': 0.1},
            'timestamps': {'offset': 1442521007.534918},
            'data_keys':
               {'offset':
                   {'dtype': 'number',
                    'source': '<descriptive string>',
                    'shape': [],
                    'units': 'mm',
                    'precision': 2}}},
         'temp_ctrl':
            {'data': {},
             'timestamps': {}
             'data_keys': {}}}
     ...}

Hints
-----

This is an experimental feature. Devices can provide information via a
``hints`` attribute that is stored here. See :ref:`hints`.

.. code-block:: python

    # excerpt of a 'descriptor' document
     {'hints':
        {'x' {'fields': ['x_readback']},
         'temp_ctrl': {'fields': ['temperature']}}
      ...}


Complete Sample
---------------

Taken together, our example 'descriptor' document looks like this.

.. code-block:: python

    # complete 'descriptor' document
    {'data_keys':
        {'temperature':
            {'dtype': 'number',
             'source': '<descriptive string>',
             'shape': [],
             'units': 'K',
             'precision': 3},
         'x_setpoint':
            {'dtype': 'number',
             'source': '<descriptive string>',
             'shape': [],
             'units': 'mm',
             'precision': 2}},
         'x_readback':
            {'dtype': 'number',
             'source': '<descriptive string>',
             'shape': [],
             'units': 'mm',
             'precision': 2}},

     'object_keys':
        {'x': ['x_setpoint', 'x_readback'],
         'temp_ctrl': ['temperature']},

     'configuration':
         {'x':
            {'data': {'offset': 0.1},
             'timestamps': {'offset': 1442521007.534918},
             'data_keys':
                {'offset':
                    {'dtype': 'number',
                     'source': '<descriptive string>',
                     'shape': [],
                     'units': 'mm',
                     'precision': 2}
          'temp_ctrl':
            {'data': {},
             'timestamps': {}
             'data_keys': {}}}
         }

     'hints':
        {'x' {'fields': ['x_readback']},
         'temp_ctrl': {'fields': ['temperature']}}

     'time': 1442521007.3438923,
     'uid': '<randomly-generated unique ID>', 
     'run_start': '<reference to the start document>'}
