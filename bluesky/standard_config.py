"""
This module creates an instance of the RunEngine and configures it. None of
this is *essential* but it is extremely useful and generally recommended.
"""
from .run_engine import RunEngine

RE = RunEngine()
register_mds(RE)  # subscribes to MDS-related callbacks
