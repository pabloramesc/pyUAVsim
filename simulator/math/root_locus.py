"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.math.transfer_function import TransferFunction


def root_locus(
    tf: TransferFunction,
    k: np.ndarray = None,
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
    k : np.ndarray, optional
        1D array of gain values for which to calculate the root locus.
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
    all_points = np.concatenate((tf.poles, tf.zeros, np.zeros(1)))
    max_center_dist = np.max(np.abs(all_points))
    real_center = np.mean(np.real(all_points))
    imag_center = np.mean(np.imag(all_points))
    real_range = np.ptp(np.real(all_points))
    imag_range = np.ptp(np.imag(all_points))
    xlim = xlim or (real_center - 2 * real_range, real_center + 1 * real_range)
    ylim = ylim or (imag_center - 1 * imag_range, imag_center + 1 * imag_range)

    if k is None:
        k = np.array([0.0, 1.0, 1e6])

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
    max_refinements = 1000
    max_iterations = max_refinements * 100
    refine_threshold = 0.01 * max_center_dist
    refine_count = 1
    for _ in range(max_iterations):
        gain1, gain2 = refined_k[refine_count - 1], refined_k[refine_count]
        roots1, roots2 = poles[refine_count - 1], poles[refine_count]

        # Check max distance between corresponding roots
        max_dist = np.max(np.abs(roots1 - roots2))

        if max_dist > refine_threshold:
            new_gain = (gain1 + gain2) / 2
            char_poly = np.polyadd(tf.den, new_gain * tf.num)
            new_roots = np.roots(char_poly)

            refined_k.insert(refine_count, new_gain)
            poles.insert(refine_count, new_roots)

        else:
            refine_count += 1
            if refine_count >= max_refinements:
                break

    poles = np.array(poles)
    refined_k = np.array(refined_k)

    if plot:
        # Create figure and axis
        plt.figure(figsize=figsize)

        # Plot the root locus
        for refine_count in range(tf.order):
            plt.plot(poles[:, refine_count].real, poles[:, refine_count].imag)

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
        plt.axis("equal")
        plt.xlim(xlim)
        plt.ylim(ylim)

        # Show the plot if requested
        if show:
            plt.show()

    return poles, refined_k
