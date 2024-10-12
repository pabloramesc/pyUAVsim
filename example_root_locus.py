"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.math.root_locus import root_locus
from simulator.math.transfer_function import TransferFunction

# Example 1: root locus with asymptotes
num = [1, 3]
den = [1, 3, 5, 1]

# Example 2: circular root locus
num = [3, 40, 40]
den = [1, 6, 45, 40]

tf = TransferFunction(num, den)

# Plot root locus with poles and zeros
gains = np.linspace(0, 40, 1000)
poles = root_locus(tf, k=gains, xlim=(-25, 0), ylim=(-15, +15))