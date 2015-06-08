Configuration
=============

When using ophyd in IPython, it is convenient to run some scripts that
import the necessary packages and define objects for each piece of hardware.
These can always be defined or redefined within the IPython session. There
is nothing special about the configuration scripts: they are just Python
code run at startup.

At NSLS-II, we use `IPython profiles <https://ipython.org/ipython-doc/dev/config/intro.html#profiles>`_ to run all the Python scripts in a given "profile"
directory at startup.

Basic Setup
-----------

Instantiate the basic scans.

.. code-block:: python

    from ophyd.userapi.scan_api import Count, AScan, DScan

    ct = Count()
    ascan = AScan()
    dscan = DScan()
    ...


Adding Hardware
---------------

Simple Detectors
^^^^^^^^^^^^^^^^

TODO

Simple Positioners
^^^^^^^^^^^^^^^^^^

TODO

Area Detectors
^^^^^^^^^^^^^^

TODO

Special Positioners
^^^^^^^^^^^^^^^^^^^

TODO
