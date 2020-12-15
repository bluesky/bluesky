===============================================
Translating Direct PyEpics Code to Bluesky Code
===============================================

.. warning:

    This section is still a work in progress.

How?
====

===========================   ======================================
interactive (blocking)        re-write for BlueSky plan()
===========================   ======================================
some.device.put("config")     yield from mv(some.device, "config")
motor.move(52)                yield from mv(motor, 52)
motor.velocity.put(5)         yield from mv(motor.velocity, 5)
===========================   ======================================


Why?
====
