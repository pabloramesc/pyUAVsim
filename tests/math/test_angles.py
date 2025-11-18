"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
import pytest
from simulator.math.angles import (
    wrap_angle_pi,
    wrap_angle_2pi,
    diff_angle_pi,
    wrap_angle,
)

PI = np.pi

# ------------------------------------------------------
# Tests for wrap_angle_pi
# ------------------------------------------------------


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (+PI, -PI),
        (-PI, -PI),
        (2 * PI, 0),
        (3 * PI, -PI),
        (-3 * PI, -PI),
    ],
)
def test_wrap_angle_pi_scalar(angle, expected):
    assert np.isclose(wrap_angle_pi(angle), expected)


def test_wrap_angle_pi_array():
    angles = np.array([0.0, +PI, -PI, 2 * PI, +3 * PI, -3 * PI])
    expected = np.array([0, -PI, -PI, 0 * PI, -1 * PI, -1 * PI])
    assert np.allclose(wrap_angle_pi(angles), expected)


# ------------------------------------------------------
# Tests for wrap_angle_2pi
# ------------------------------------------------------


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0, 2 * PI),
        (PI, PI),
        (2 * PI, 2 * PI),
        (3 * PI, PI),
        (-PI, PI),
    ],
)
def test_wrap_angle_2pi_scalar(angle, expected):
    assert np.isclose(wrap_angle_2pi(angle), expected)


def test_wrap_angle_2pi_array():
    angles = np.array([0.0 * PI, PI, 2 * PI, 3 * PI, -PI])
    expected = np.array([2 * PI, PI, 2 * PI, 1 * PI, +PI])
    assert np.allclose(wrap_angle_2pi(angles), expected)


# -------------------------
# Tests for diff_angle_pi
# -------------------------
@pytest.mark.parametrize(
    "a1, a2, expected",
    [
        (0.0, 0.0, 0.0),
        (+PI, +PI, 0.0),
        (0.0, +PI, -PI),
        (+PI, 0.0, -PI),
        (+PI, -PI, 0.0),
        (-PI, +PI, 0.0),
        (3 / 2 * PI, 0, -1 / 2 * PI),
        (0, 3 / 2 * PI, +1 / 2 * PI),
    ],
)
def test_diff_angle_pi_scalar(a1, a2, expected):
    diff = diff_angle_pi(a1, a2)
    assert np.isclose(diff, expected)


def test_diff_angle_pi_array():
    angle1 = np.array([0.0, +PI, 0.0, +PI, +PI, +3 / 2 * PI])
    angle2 = np.array([0.0, +PI, +PI, 0.0, -PI, 0])
    expected = np.array([0, 0.0, -PI, -PI, 0.0, -1 / 2 * PI])
    diffs = diff_angle_pi(angle1, angle2)
    assert np.allclose(diffs, expected)


# -------------------------
# Tests for wrap_angle
# -------------------------
@pytest.mark.parametrize(
    "a1, a2",
    [
        (0.0, 0.0),
        (+PI, +PI),
        (0.0, +PI),
        (+PI, 0.0),
        (+PI, -PI),
        (-PI, +PI),
        (3 / 2 * PI, 0),
        (0, 3 / 2 * PI),
    ],
)
def test_wrap_angle_scalar(a1, a2):
    w1 = wrap_angle(a1, a2)
    diff = a2 - w1
    assert -PI <= diff <= +PI


def test_wrap_angle_array():
    a1 = np.array([0.0, +PI, 0.0, +PI, +PI, +3 / 2 * PI])
    a2 = np.array([0.0, +PI, +PI, 0.0, -PI, 0])
    w1 = wrap_angle(a1, a2)
    diffs = a2 - w1
    assert np.all((-PI <= diffs) & (diffs <= +PI))
