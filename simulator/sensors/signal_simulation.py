"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np


def extend_data(t: np.ndarray, data: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Interpolate random spaced data to get uniformally sapaced data.

    Arguments
    ---------
    t : np.ndarray
        N size array with time of each sample.

    data : np.ndarray
        N-by-D array with N samples of D dimensions of original data.

    sample_rate : float
        Desired frequency to space the interpolated data.

    Returns
    -------
    np.ndarray
        M-by-D array with M samples of D dimensions of interpolated data.

    """
    t_min = np.amin(t)
    t_max = np.amax(t)
    dt = 1.0 / sample_rate
    extend_t = np.arange(t_min, t_max + dt, dt)
    extend_data = np.interp(extend_t, t, data)
    return extend_data


def saturate(val: np.ndarray, min_val: float, max_val: float):
    """
    Saturated input data to simulate measurement saturation.

    Arguments
    ---------
    val : np.ndarray
        Input data to saturate.

    min_val : float
        Minimum value of the saturated data to output.

    max_val : float
        Maximum value of the saturated data to output.

    Returns
    -------
    np.ndarray
        N array with saturated data.

    """
    sat_val = np.clip(val, min_val, max_val)
    return sat_val


def digitalize(val: np.ndarray, full_scale: float, bits: int) -> np.ndarray:
    """
    Digitalize data to simulate quantization of and ADC (Analog-to-Digital-Converter).

    Arguments
    ---------
    val : np.ndarray
        Input data to digitalize.

    full_scale : float
        Amplitude of the output data. Maximum output will be +full_scale while minimum output will be -full_scale.

    bits : integer
        Number of bits of the converter.

    Returns
    -------
    np.ndarray
        N array with digitalized data.

    """
    # transfer_function = np.linspace(-full_scale, full_scale, 2**bits)
    # bin_indices = np.digitize(val, transfer_function, right=True)
    # return transfer_function[bin_indices]
    resolution = 2 * full_scale / 2**bits
    bin_index: np.ndarray = (val + full_scale) / resolution
    if isinstance(bin_index, float):
        bin_index = int(bin_index)
    if isinstance(bin_index, np.ndarray):
        bin_index = bin_index.astype(int)
    return bin_index * resolution - full_scale
