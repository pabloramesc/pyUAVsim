"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Tuple

import numpy as np


def check_array(
    a: np.ndarray, shape: Tuple[int, ...] = None, name: str = "Array"
) -> None:
    # Check numpy array type
    if not isinstance(a, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, but got {type(a)}.")
    # Check numpy array shape
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} must have shape: {shape}, but got {a.shape}.")
