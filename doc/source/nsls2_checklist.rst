NSLS-II Beamline Configuration Checklist
========================================

The following are prerequisites for reliable data collection.

* The clocks of all beamline computers should be synchronized to the "ring
  clock." On UNIX machines, this is likely already done by the system
  administrator. On Windows machines, this can be done in time settings.
  The Network Time Protcol server (NTP server) is ``time.cs.nsls2.local``.
  By default, syncing is not performed very often. It is best to configure
  the settings so that syncing is performed every hour.
