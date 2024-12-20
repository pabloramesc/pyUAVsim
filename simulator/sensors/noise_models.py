"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np


def get_white_noise(Nd: float, fs: float, nlen: int = 1) -> float | np.ndarray:
    """Generate white noise of desired characteristics.

    Parameters
    ----------
    Nd : float
        Noise density parameter
    fs : float
        Sample rate in Hz
    nlen : int, optional
        Number of samples to return, by default 1

    Returns
    -------
    float or np.ndarray
        Simulated white noise sample or array

    Notes
    -----
    WN_k = N / sqrt(dt) * wk

    where:
        - WN_k is white noise in instant k
        - N is noise density parameter of allan variance analysis
        - dt is de sampling period (cte, variable, average or ...)
        - wk is normal estandar gaussian distribued noise N(0,1) in instant k

    """
    if nlen == 1:
        wn = Nd * np.sqrt(fs) * np.random.normal(0, 1)
    elif nlen > 1:
        wn = Nd * np.sqrt(fs) * np.random.normal(0, 1, nlen)
    return wn


def get_pink_noise(
    Bi: float, Tc: float, fs: float, pn_prev: float = 0.0, nlen: int = 1
) -> float | np.ndarray:
    """Generate pink (flicker) noise of desired characteristics.
    It is generated by using a gauss-markov first order process.

    Parameters
    ----------
    Bi : float
        Bias instability parameter
    Tc : float
        Correlation time parameter
    fs : float
        Sample rate in Hz
    pn_prev : float, optional
        Previous pink noise sample for iterative simulation, by default 0.0
    nlen : int, optional
        Number of samples o return, by default 1

    Returns
    -------
    float or np.ndarray
        Simulated pink noise sample or array

    Notes
    -----
    PN_k = (1 - beta*dt) * PN_k-1 + B * sqrt(1 - exp(-2*dt/Tc)) * wk

    where:
        - PN_k is pink noise in instant k
        - PN_k-1 is pink noise in the previous instant k
        - B is bias instability parameter of allan variance analysis
        - Tc is correlation time asociated with the bias instability
        - beta is the inverse of Tc (beta = 1/Tc)
        - dt is de sampling period (cte, variable, average or ...)
        - wk is normal estandar gaussian distribued noise N(0,1) in instant k

    """
    dt = 1.0 / fs
    beta = 1.0 / Tc
    if nlen == 1:
        pn = (1.0 - beta * dt) * pn_prev + Bi * np.sqrt(
            1.0 - np.exp(-2.0 * dt / Tc)
        ) * np.random.normal(0, 1)
    elif nlen > 1:
        pn = np.zeros(nlen)
        for k in range(nlen):
            pn[k] = (1.0 - beta * dt) * pn_prev + Bi * np.sqrt(
                1.0 - np.exp(-2.0 * dt / Tc)
            ) * np.random.normal(0, 1)
            pn_prev = pn[k]
    return pn


def get_brown_noise(
    Rw: float, fs: float, bn_prev: float = 0.0, nlen: int = 1
) -> float | np.ndarray:
    """Generate brown noise of desired characteristics.
    It is generated by integrating white noise.

    Parameters
    ----------
    Rw : float
        Random walk parameter
    fs : float
        Sample rate in Hz
    bn_prev : float, optional
        Previous brown noise sample for iterative simulation, by default 0.0
    nlen : int, optional
        Number of samples to return, by default 1

    Returns
    -------
    float or np.ndarray
        Simulated brown noise sample or array

    Notes
    -----
    BN_k = BN_k-1 + K * sqrt(dt) * wk

    where:
        - BN_k is brown noise in instant k
        - BN_k-1 is brown noise in the previous instant k
        - K is random walk parameter of allan variance analysis
        - dt is de sampling period (cte, variable, average or ...)
        - wk is normal estandar gaussian distribued noise N(0,1) in instant k
    """
    if nlen == 1:
        bn = bn_prev + Rw / np.sqrt(fs) * np.random.normal(0, 1)
    elif nlen > 1:
        bn = bn_prev + np.cumsum(Rw / np.sqrt(fs) * np.random.normal(0, 1, nlen))
    return bn
