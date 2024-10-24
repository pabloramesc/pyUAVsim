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
    Calculate and plot the root locus for a given transfer function.

    This method applies adaptative gain refinement to improve the quality
    of the root locus plot by inserting new gain values between poles to
    capture transitions more accurately.

    Parameters
    ----------
    tf : TransferFunction
        The transfer function object.
    k : array_like, optional
        Initial gain values for which to calculate the root locus.
        If None, a default range from 0 to 10 is generated.
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
        The shape of the array is (K, N), where K is the number of gain values
        generated and N is the order of the transfer function.
    """
    # Calculate real and imag values range for poles and zeros
    all_points = np.concatenate((tf.poles, tf.zeros, np.zeros(1)))
    max_center_dist = np.max(np.abs(all_points))
    real_mean = np.mean(np.real(all_points))
    imag_mean = np.mean(np.imag(all_points))
    real_range = np.ptp(np.real(all_points))
    real_range = real_range if real_range > 0.0 else 1.0
    imag_range = np.ptp(np.imag(all_points))
    imag_range = imag_range if imag_range > 0.0 else real_range

    # Calculate default axis limits
    xlim = xlim or (real_mean - 2 * real_range, real_mean + 1 * real_range)
    ylim = ylim or (imag_mean - 1 * imag_range, imag_mean + 1 * imag_range)

    if k is None:
        k = [0.0, 1.0, 1e12]

    # Calculate poles for initial k values
    poles = []
    for gain in k:
        # Calculate the characteristic polynomial coefficients
        # Coefficients for 1 + K * G(s)
        char_poly = np.polyadd(tf.den, gain * tf.num)
        # Calculate the roots of the characteristic polynomial
        roots = np.roots(char_poly)
        poles.append(roots)

    # Refinement process
    refined_k = list(k)
    max_refinements = 100
    max_iterations = max_refinements * 100
    refine_threshold = 0.01 * max_center_dist
    refine_index = 1 # position to refine
    for _ in range(max_iterations):
        gain1, gain2 = refined_k[refine_index-1], refined_k[refine_index]
        roots1, roots2 = poles[refine_index-1], poles[refine_index]

        # Check max distance between corresponding roots
        max_dist = np.max(np.abs(roots1 - roots2))

        # If distance between roots is bigger the threshold, insert new gain in the middle
        if max_dist > refine_threshold:
            new_gain = (gain1 + gain2) / 2
            char_poly = np.polyadd(tf.den, new_gain * tf.num)
            new_roots = np.roots(char_poly)

            refined_k.insert(refine_index, new_gain)
            poles.insert(refine_index, new_roots)

        else:
            refine_index += 1

        # If all roots are outside bounds, skip iteration and remove the last root
        # Note: keep at least one root out bounds to plot lines correctly
        roots1_in_bounds = np.any(_check_roots_in_bounds(poles[-2], xlim, ylim))
        roots2_in_bounds = np.any(_check_roots_in_bounds(poles[-1], xlim, ylim))
        if not roots1_in_bounds and not roots2_in_bounds:
            refined_k.pop(-1)
            poles.pop(-1)

        if refine_index >= len(refined_k):
            break

    poles = np.array(poles)
    refined_k = np.array(refined_k)

    if plot:
        # Create figure and axis
        plt.figure(figsize=figsize)

        # Plot the root locus
        for refine_index in range(tf.order):
            plt.plot(poles[:, refine_index].real, poles[:, refine_index].imag)

        # Plot the poles and zeros
        if len(tf.poles) > 0:
            plt.scatter(
                tf.poles.real,
                tf.poles.imag,
                marker="x",
                color=pole_color,
                s=marker_size,
                label="Poles",
            )
        if len(tf.zeros) > 0:
            plt.scatter(
                tf.zeros.real,
                tf.zeros.imag,
                marker="o",
                color=zero_color,
                s=marker_size,
                label="Zeros",
            )

        # Add grid, legend, and labels
        plt.axhline(0, color="black", lw=0.5, ls="--")
        plt.axvline(0, color="black", lw=0.5, ls="--")
        plt.title("Root Locus")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.legend()
        plt.grid()
        # plt.axis("equal")
        plt.xlim(xlim)
        plt.ylim(ylim)

        # Show the plot if requested
        if show:
            plt.show()

    return poles, refined_k


def _check_roots_in_bounds(roots: ArrayLike, xlim: tuple, ylim: tuple) -> list:
    """
    Check if each root is within the specified x and y limits. Return a list.
    """
    real_in_bounds = (np.real(roots) >= xlim[0]) & (np.real(roots) <= xlim[1])
    imag_in_bounds = (np.imag(roots) >= ylim[0]) & (np.imag(roots) <= ylim[1])
    return (real_in_bounds & imag_in_bounds)