"""Unit tests for `Suspension` itself; behaviour through a RunEngine is in `test_suspenders.py`."""

import asyncio

from bluesky import Msg
from bluesky.suspension import Suspension, join_justifications

from .utils import _parked


def test_a_child_suspension_is_tripped_whenever_its_parent_is():

    async def check():
        loop = asyncio.get_running_loop()
        parent = Suspension(loop)
        child = Suspension(loop, parent=parent)

        parent.trip("beam", "beam is down")
        # Held up by its parent.
        assert child.reasons
        assert join_justifications(child.reasons) == "beam is down"

        child.trip("shutter", "shutter is closed")
        parent.recover("beam")
        # Still holding its own reason.
        assert child.reasons
        assert not parent.reasons

        child.recover("shutter")
        assert not child.reasons

    asyncio.run(check())


def test_a_trip_between_building_the_plan_and_running_it_still_holds():
    from bluesky.plan_session import PlanSession

    steps = []

    def plan():
        yield Msg("checkpoint")
        for _ in range(3):
            steps.append("step")
            yield Msg("sleep", None, 0.05)

    async def main():
        session = PlanSession()
        runner = session.start(plan())
        # Nothing had tripped when this was built.
        assert not runner._suspension.reasons

        session._suspension.trip("beam", "beam is down")
        task = asyncio.ensure_future(runner)
        await _parked()
        held = list(steps)

        session._suspension.recover("beam")
        await asyncio.wait_for(task, timeout=10)
        return held

    ran_while_tripped = asyncio.run(main())

    assert ran_while_tripped == []
    assert steps == ["step"] * 3
