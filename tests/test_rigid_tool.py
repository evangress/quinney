"""Reaction force on a velocity-driven rigid tool."""

import numpy as np
import pytest

from quinney.kratos_adapter.rigid_tool import reaction_force


class TestReactionForce:
    """The tool is driven at constant speed, so the workpiece cannot slow it.

    The impulse needed each step to restore the prescribed velocity *is* the
    force the workpiece exerts on the tool — which is the cutting force. That
    makes force a by-product of driving the tool, with nothing extra to
    instrument.
    """

    def test_an_undisturbed_tool_feels_no_force(self):
        force = reaction_force(
            mass=np.array([1.0, 1.0]),
            velocity=np.array([[-20.0, 0.0], [-20.0, 0.0]]),
            target_velocity=(-20.0, 0.0),
            time_step_s=1e-8,
        )
        assert force == pytest.approx([0.0, 0.0])

    def test_a_decelerated_tool_feels_force_opposing_its_motion(self):
        # Tool driven in -x, slowed by the cut: the workpiece pushes back in +x.
        force = reaction_force(
            mass=np.array([2.0]),
            velocity=np.array([[-19.0, 0.0]]),
            target_velocity=(-20.0, 0.0),
            time_step_s=1e-3,
        )
        assert force[0] == pytest.approx(2.0 * 1.0 / 1e-3)
        assert force[0] > 0.0

    def test_sums_the_impulse_over_every_tool_point(self):
        force = reaction_force(
            mass=np.array([1.0, 3.0]),
            velocity=np.array([[-19.0, 0.0], [-19.0, 0.0]]),
            target_velocity=(-20.0, 0.0),
            time_step_s=1e-3,
        )
        assert force[0] == pytest.approx(4.0 * 1.0 / 1e-3)

    def test_reports_thrust_as_well_as_cutting_force(self):
        # Thrust is the y component. A tool that picked up downward velocity was
        # pushed downward, so the force on it is negative — the sign follows the
        # direction the tool was disturbed, not the direction of the correction.
        force = reaction_force(
            mass=np.array([1.0]),
            velocity=np.array([[-20.0, -0.5]]),
            target_velocity=(-20.0, 0.0),
            time_step_s=1e-3,
        )
        assert force[1] == pytest.approx(-0.5 / 1e-3)

    def test_a_tool_pushed_away_from_the_cut_reports_positive_thrust(self):
        # Cutting the top surface pushes the tool up, off the workpiece.
        force = reaction_force(
            mass=np.array([1.0]),
            velocity=np.array([[-20.0, 0.5]]),
            target_velocity=(-20.0, 0.0),
            time_step_s=1e-3,
        )
        assert force[1] == pytest.approx(0.5 / 1e-3)

    def test_rejects_a_non_positive_time_step(self):
        with pytest.raises(ValueError):
            reaction_force(
                mass=np.array([1.0]),
                velocity=np.array([[-20.0, 0.0]]),
                target_velocity=(-20.0, 0.0),
                time_step_s=0.0,
            )
