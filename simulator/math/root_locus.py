"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from simulator.math.transfer_function import TransferFunction


def root_locus(
    tf: TransferFunction,
    k: ArrayLike = None,
    figsize: tuple = None,
    xlim: tuple = None,
    ylim: tuple = None,
    pole_color: str = "red",
    zero_color: str = "blue",
    marker_size: int = 20,
    plot: bool = True,
    show: bool = True,
):
    """
    Calculate and plot the root locus for a given transfer function with adaptive gain refinement.

    Parameters
    ----------
    tf : TransferFunction
        The transfer function object.
    k : array_like, optional
        Initial gain values for which to calculate the root locus.
        If None, a default range from 0 to 1e12 is generated.
    figsize : tuple, optional
        Size of the figure (width, height).
    xlim : tuple, optional
        x-axis limits (xmin, xmax). If None, limits will be auto-scaled.
    ylim : tuple, optional
        y-axis limits (ymin, ymax). If None, limits will be auto-scaled.
    pole_color : str, optional
        Color for the poles in the plot.
    zero_color : str, optional
        Color for the zeros in the plot.
    marker_size : int, optional
        Size of the marker for poles and zeros.
    plot : bool, optional
        If True, plot the root locus. If False, just compute the values.
    show : bool, optional
        If True, display the plot. If False, do not show the plot.

    Returns
    -------
    np.ndarray
        2D array containing the calculated poles for each gain value.
    np.ndarray
        The corresponding refined gain values.
    """
    # Set initial gain values if not provided
    if k is None:
        k = [0.0, 1.0, 1e12]

    # Get range and mean for auto-scaling limits
    xlim, ylim = _compute_axis_limits(tf.poles, tf.zeros, xlim, ylim)

    # Calculate poles for initial k values
    poles = [_calculate_roots(tf, gain) for gain in k]

    # Refine gain values and poles for better accuracy
    poles, gains = _refine_gains_and_poles(tf, k, poles, xlim, ylim)

    # Plot root locus if required
    if plot:
        _plot_root_locus(
            tf, poles, xlim, ylim, pole_color, zero_color, marker_size, figsize, show
        )

    return np.array(poles), np.array(gains)


def _compute_axis_limits(
    poles: ArrayLike, zeros: ArrayLike, xlim: tuple, ylim: tuple
) -> tuple:
    """
    Compute the default axis limits for the root locus plot if not provided.
    """
    all_points = np.concatenate((poles, zeros, np.zeros(1)))

    real_range = np.ptp(np.real(all_points))
    imag_range = np.ptp(np.imag(all_points))

    real_range = real_range if real_range > 0.0 else 1.0
    imag_range = imag_range if imag_range > 0.0 else real_range

    real_mean = np.mean(np.real(all_points))
    imag_mean = np.mean(np.imag(all_points))

    xlim = xlim or (real_mean - 2 * real_range, real_mean + 1 * real_range)
    ylim = ylim or (imag_mean - imag_range, imag_mean + imag_range)

    return xlim, ylim


def _calculate_roots(tf: TransferFunction, gain: float) -> np.ndarray:
    """
    Calculate the roots of the characteristic equation for a given gain.
    """
    char_poly = np.polyadd(tf.den, gain * tf.num)
    return np.roots(char_poly)


def _refine_gains_and_poles(
    tf: TransferFunction,
    gains: list,
    poles: list,
    xlim: tuple,
    ylim: tuple,
    max_iterations: int = 10000,
    refine_threshold: float = 0.01,
) -> tuple:
    """
    Refine gain values and poles for better root locus accuracy.
    """
    refined_gains = list(gains)
    refined_poles = list(poles)
    refine_index = 1
    max_center_dist = np.max(np.abs(np.concatenate((tf.poles, tf.zeros, np.zeros(1)))))
    refine_dist = refine_threshold * max_center_dist

    for _ in range(max_iterations):
        if refine_index >= len(refined_gains):
            break

        gain1, gain2 = refined_gains[refine_index - 1], refined_gains[refine_index]
        roots1, roots2 = refined_poles[refine_index - 1], refined_poles[refine_index]

        max_dist = np.max(np.abs(roots1 - roots2))

        if max_dist > refine_dist:
            new_gain = (gain1 + gain2) / 2
            new_roots = _calculate_roots(tf, new_gain)

            refined_gains.insert(refine_index, new_gain)
            refined_poles.insert(refine_index, new_roots)
        else:
            refine_index += 1

        # Remove roots with all poles out of bounds
        roots1_in_bounds = _get_roots_in_bounds(refined_poles[-3], xlim, ylim)
        roots2_in_bounds = _get_roots_in_bounds(refined_poles[-1], xlim, ylim)
        if (roots1_in_bounds.size == 0) and (roots2_in_bounds.size == 0):
            refined_gains.pop(-1)
            refined_poles.pop(-1)

        # Remove roots inside bounds with very close poles
        if (roots1_in_bounds.size > 0) and (roots2_in_bounds.size > 0):
            max_dist = np.max(np.abs(roots1_in_bounds - roots2_in_bounds))
            if max_dist < refine_dist:
                refined_gains.pop(-2)
                refined_poles.pop(-2)

    return refined_poles, refined_gains


def _plot_root_locus(
    tf, poles, xlim, ylim, pole_color, zero_color, marker_size, figsize, show
):
    """
    Plot the root locus based on the refined poles and gains.
    """
    plt.figure(figsize=figsize)

    poles = np.array(poles)
    for i in range(tf.order):
        plt.plot(poles[:, i].real, poles[:, i].imag)

    plt.scatter(
        tf.poles.real,
        tf.poles.imag,
        marker="x",
        color=pole_color,
        s=marker_size,
        label="Poles",
    )
    plt.scatter(
        tf.zeros.real,
        tf.zeros.imag,
        marker="o",
        color=zero_color,
        s=marker_size,
        label="Zeros",
    )

    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
    plt.title("Root Locus")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)

    if show:
        plt.show()


def _get_roots_in_bounds(roots: ArrayLike, xlim: tuple, ylim: tuple) -> np.ndarray:
    """
    Return each root which is inside the specified x and y limits.
    """
    real_in_bounds = (np.real(roots) >= xlim[0]) & (np.real(roots) <= xlim[1])
    imag_in_bounds = (np.imag(roots) >= ylim[0]) & (np.imag(roots) <= ylim[1])
    return np.array(roots)[real_in_bounds & imag_in_bounds]
