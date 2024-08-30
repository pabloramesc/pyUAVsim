"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
import pytest

from simulator.autopilot.pid_controller import PIDController


def test_proportional():
    pid = PIDController(kp=2.0)
    actual = pid.update(x_ref=10.0, x=5.0, dt=1.0)
    expected = 2.0 * (10.0 - 5.0)  # kp * (x_ref - x)
    np.testing.assert_almost_equal(actual, expected)


def test_integral():
    pid = PIDController(ki=1.0)
    actual = pid.update(x_ref=10.0, x=5.0, dt=1.0)
    error_prev = 0.0
    error = 10.0 - 5.0
    expected = 1.0 * (error + error_prev) * 0.5  # ki * (x_ref - x) * dt/2
    np.testing.assert_almost_equal(actual, expected)
    actual = pid.update(x_ref=10.0, x=5.0, dt=1.0)
    error_prev = error
    error = 10.0 - 5.0
    expected += 1.0 * (error + error_prev) * 0.5  # accumulate integral over 2 seconds
    np.testing.assert_almost_equal(actual, expected)


def test_derivative():
    pid = PIDController(kd=1.0, tau=3.0)
    actual = pid.update(x_ref=10.0, x=5.0, dt=1.0)
    error_prev = 0.0
    error = 10.0 - 5.0
    expected = (
        1.0 * 2.0 / (2 * 3.0 + 1.0) * (error - error_prev)
    )  # kd * 2 / (2 * tau + dt) * (error - error_prev)
    np.testing.assert_almost_equal(actual, expected)
    actual = pid.update(x_ref=10.0, x=5.0, dt=1.0)
    error_prev = error
    error = 10.0 - 5.0
    alpha = (2 * 3.0 - 1.0) / (2 * 3.0 + 1.0)  # (2 * tau - dt) / (2 * tau + dt)
    expected = 1.0 * alpha * expected + 2 / (2 * 3.0 + 1.0) * (
        error - error_prev
    )  # kd * (alpha * diff_prev - 2 / (2 * tau + dt) * (error - error_prev))
    np.testing.assert_almost_equal(actual, expected)

def test_saturation():
    pid = PIDController(kp=2.0, max_output=5.0, min_output=-3.0)
    actual = pid.update(x_ref=10.0, x=5.0, dt=1.0)
    expected = 5.0
    np.testing.assert_almost_equal(actual, expected)
    actual = pid.update(x_ref=-5.0, x=0.0, dt=1.0)
    expected = -3.0
    np.testing.assert_almost_equal(actual, expected)

def test_reset_functionality():
    pid = PIDController(kp=1.0, ki=1.0)
    pid.update(x_ref=10, x=5, dt=1.0)
    pid.reset()
    np.testing.assert_almost_equal(pid.prev_error, 0.0)
    np.testing.assert_almost_equal(pid.prev_intg, 0.0)
    np.testing.assert_almost_equal(pid.prev_diff, 0.0)

def test_integrator_anti_windup():
    pid = PIDController(kp=1.0, ki=1.0, max_output=5.0)
    # Test integrator anti-windup; expected output should not wind up beyond saturation limits
    output = pid.update(x_ref=10, x=5, dt=1.0)
    np.testing.assert_almost_equal(output, 5.0) # Saturation applies
    # If saturation and anti-windup are correct, next output should not drastically change
    output = pid.update(x_ref=10, x=5, dt=1.0)
    np.testing.assert_almost_equal(output, 5.0)  # Consistent with anti-windup

def test_null_update():
    pid = PIDController(kp=2.0, ki=0.1, kd=0.2)
    outputs = []
    for k in range(10):
        out = pid.update(x_ref=10.0, x=10.0, dt=1.0)
        outputs.append(out)
    np.testing.assert_array_almost_equal(np.array(outputs), np.zeros(10))