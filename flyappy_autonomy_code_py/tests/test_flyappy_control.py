import numpy as np
from flyappy_autonomy_code.core.flyappy_control import ProportionalController, StateFeedbackController


# -----------------------------
# ProportionalController Tests
# -----------------------------

def test_proportional_controller_zero_error():
    """
    Scenario: 
        - Setpoint equals value (no error).

    Expectation: 
        - Output should be zero
    """
    controller = ProportionalController(kp=4.0, acc_max=3.0)

    acc = controller.control_step(
        setpoint=2.0,
        velocity=2.0,
        dt=0.1
    )

    assert acc == 0.0


def test_proportional_controller_linear_region():
    """
    Scenario: 
        - Small error within output limits.

    Expectation:
        - Output equals kp * error.
    """
    controller = ProportionalController(kp=2.0, acc_max=10.0)

    acc = controller.control_step(setpoint=3.0, velocity=1.0, dt=0.1)
    # error = 2.0 → acc = 4.0
    assert acc == 4.0


def test_proportional_controller_saturation():
    """
    Scenario:
        - Large error exceeding output limits.

    Expectation:
        - Output is clipped to +acc_max.
    """
    controller = ProportionalController(kp=10.0, acc_max=3.0)

    acc = controller.control_step(setpoint=10.0, velocity=0.0, dt=0.1)
    assert acc == 3.0

# -----------------------------
# StateFeedbackController Tests
# -----------------------------

def test_state_feedback_zero_error():
    """
    Scenario:
        - Current position matches setpoint and velocity is zero.

    Expectation:
        - Output should be zero.
    """
    controller = StateFeedbackController(kp=4.0, kd=10.0)

    acc = controller.control_step(setpoint=5.0, position=5.0, velocity=0.0)
    assert acc == 0.0


def test_state_feedback_velocity_clipping():
    """
    Scenario:
        - Large position error produces desired velocity exceeding vel_max.

    Expectation:
        - Output reflects clipped velocity value.
    """
    controller = StateFeedbackController(kp=100.0, kd=1.0, vel_max=2.0, acc_max=10.0)
    
    acc = controller.control_step(setpoint=10.0, position=0.0, velocity=0.0)
    
    # vel_des clipped to 2.0 → acc = 2.0
    assert acc == 2.0


def test_state_feedback_acceleration_saturation():
    """
    Scenario: 
        - Large velocity error produces acceleration exceeding acc_max.
    
    Expectation: 
        - Acceleration output is clipped to acc_max.
    """
    controller = StateFeedbackController(kp=10.0, kd=10.0, vel_max=10.0, acc_max=5.0)
    
    acc = controller.control_step(setpoint=10.0, position=0.0, velocity=-10.0)
    assert acc == 5.0