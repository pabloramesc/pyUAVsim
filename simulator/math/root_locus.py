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
    zero_color: str = "green",
    locus_color: str = "blue",
    marker_size: int = 10,
    plot: bool = True,
    show: bool = True,
):
    """
    Calculate and optionally plot the root locus for a given transfer function.

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
    locus_color : str, optional
        Color for the root locus lines in the plot.
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
    if k is None:
        k = np.linspace(0, 10, 100)

    poles = np.zeros((len(k), tf.order), dtype=complex)

    for i, gain in enumerate(k):
        # Calculate the characteristic polynomial coefficients
        # Coefficients for 1 + K * G(s)
        char_poly = np.polyadd(tf.den, gain * tf.num)
        # Calculate the roots of the characteristic polynomial
        poles[i, :] = np.roots(char_poly)

    if plot:
        # Create figure and axis
        plt.figure(figsize=figsize)

        # Plot the root locus
        for i in range(tf.order):
            plt.plot(poles[:, i].real, poles[:, i].imag, locus_color)

        # Plot the poles and zeros
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

        # Set x and y limits
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

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

    return poles
