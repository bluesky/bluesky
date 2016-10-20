def test_broker_examples():
    from bluesky.examples import det_2d, motor
    from .utils import setup_test_run_engine
    from bluesky.plans import DeltaScanPlan
    RE = setup_test_run_engine()
    plan = DeltaScanPlan([det_2d], motor, -1, 1, 10)
    RE(plan)
