"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
import pytest

from simulator.math.rotation import (
    rot_matrix_axis,
    rot_matrix_zyx,
    rot_matrix_quat,
    rot_matrix_wind,
    rotate,
    rotate_points,
    multi_rotation,
    ned2xyz,
    euler2quat,
    quat2euler,
    euler_kinematics,
    quaternion_kinematics,
)


def test_rot_matrix_axis():
    """Rotate x axis around z axis by 90 degrees
    """
    axis = np.array([0, 0, 1]) # z axis
    angle = np.pi / 2 # 90 degress
    Rot = rot_matrix_axis(axis, angle)
    actual = Rot @ np.array([1, 0, 0]) # rotate x axis
    expected = np.array([0, 1, 0]) # result is y axis
    np.testing.assert_array_almost_equal(actual, expected)


def test_rot_matrix_zyx():
    euler = np.array([np.pi / 2, 0, 0])
    Rot = rot_matrix_zyx(euler)
    actual = Rot @ np.array([0, 0, 1])
    expected = np.array([0, 1, 0])
    np.testing.assert_array_almost_equal(actual, expected)

def test_rot_matrix_quat():
    quat = np.array([1.0, 1.0, 0.0, 0.0]) # roll = 90 deg
    quat = quat / np.linalg.norm(quat)
    Rot = rot_matrix_quat(quat)
    actual = Rot @ np.array([0, 0, 1])
    expected = np.array([0, 1, 0])
    np.testing.assert_array_almost_equal(actual, expected)

def test_rot_matrix_wind():
    alpha = np.pi / 4
    beta = np.pi / 4
    actual = rot_matrix_wind(alpha, beta)
    expected = np.array(
        [
            [0.5, -0.5, -np.sqrt(2) / 2],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [0.5, -0.5, np.sqrt(2) / 2],
        ]
    )
    np.testing.assert_array_almost_equal(actual, expected)


def test_rotate():
    R = np.eye(3)
    origin = np.zeros(3)
    point = np.array([1, 2, 3])
    actual = rotate(R, origin, point)
    expected = point
    np.testing.assert_array_equal(actual, expected)


def test_rotate_points():
    R = np.eye(3)
    origin = np.zeros(3)
    points = np.array([[1, 2, 3], [4, 5, 6]])
    actual = rotate_points(R, origin, points)
    expected = points
    np.testing.assert_array_equal(actual, expected)


def test_multi_rotation():
    angles = np.zeros((2, 3))
    points = np.array([[1, 2, 3], [4, 5, 6]])
    actual = multi_rotation(angles, points)
    expected = points
    np.testing.assert_array_equal(actual, expected)


def test_ned2xyz():
    ned_coords = np.array([1, 2, -3])
    actual = ned2xyz(ned_coords)
    expected = np.array([2, 1, 3])
    np.testing.assert_array_equal(actual, expected)


def test_euler2quat():
    euler = np.array([np.pi/2, 0, 0])
    actual = euler2quat(euler)
    expected = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
    np.testing.assert_array_almost_equal(actual, expected)


def test_quat2euler():
    quat = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
    actual = quat2euler(quat)
    expected = np.array([np.pi/2, 0, 0])
    np.testing.assert_array_almost_equal(actual, expected)


def test_euler_kinematics():
    omega = np.array([0.1, 0.2, 0.3])
    euler = np.zeros(3)
    R_euler = euler_kinematics(euler)
    actual = R_euler @ omega
    expected = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(actual, expected)

def test_quaternion_kinematics():
    omega = np.array([0.1, 0.2, 0.3])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    Omega = quaternion_kinematics(omega)
    actual = 0.5 * Omega @ quat
    expected = 0.5 * np.array([0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(actual, expected)

if __name__ == "__main__":
    pytest.main()
