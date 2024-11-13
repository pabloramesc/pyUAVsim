"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.math.root_locus import root_locus
from simulator.math.transfer_function import TransferFunction

# Example 1
num = [1]
den = [1, 3, 0]

# Example 2
num = [1]
den = [1, 5, 6, 0]

# Example 3
num = [1, 3]
den = [1, -1, -2]

# Example 4
num = [1, 3]
den = [1, 3, 5, 1]


tf = TransferFunction(num, den)

# Plot root locus with poles and zeros
poles, gains = root_locus(tf)

print(f"{len(gains)} gains calculated\n")
