.. _hardware_interface_packages:

Hardware Interface Packages
===========================

The Bluesky library does not provide direct support for communicating with real hardware.
Instead, we define a high-level abstraction: the :ref:`Bluesky Hardware Interface <hardware_interface>`.
This allows different experimentalists to use different hardware control systems.

The following packages provide support for real hardware communication from Bluesky:

=============  ================================================================================
Ophyd_         EPICS_ integration for Bluesky. Reference implementation for hardware interface.
Instrbuilder_  Lightweight package with a focus on SCPI_.
Ophyd-Tango_   Tango_ integration for Bluesky. Incomplete and experimental early work.
pycertifspec_  Communication with SPEC_ instruments.
yaqc-bluesky_  yaq_ integration for Bluesky.
=============  ================================================================================

Importantly, you may mix hardware interfaces from multiple different packages within the same RunEngine.
Please note that the above packages are developed and maintained separately from Bluesky itself.

Are you maintaining a Python Package which provides hardware communication functionality?
The :ref:`Bluesky Hardware Interface <hardware_interface>` is a simple set of attributes and methods that can easily be added to your existing classes.
Please consider supporting our interface specification to unlock the full capabilities of the Bluesky ecosystem for your supported hardware.
Let us know if you add Bluesky support so we can add you to the above list.

.. _Instrbuilder: https://lucask07.github.io/instrbuilder/build/html/
.. _SCPI: https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments
.. _Ophyd: https://blueskyproject.io/ophyd/
.. _EPICS: https://epics-controls.org/
.. _Ophyd-Tango: https://github.com/bluesky/ophyd-tango
.. _pycertifspec: https://github.com/SEBv15/pycertifspec
.. _SPEC: https://www.certif.com/content/spec/
.. _Tango: https://www.tango-controls.org/
.. _yaqc-bluesky: https://github.com/bluesky/yaqc-bluesky
.. _yaq: https://yaq.fyi/
