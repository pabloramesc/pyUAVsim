"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulator.aircraft import AircraftState
from simulator.math.rotation import ned2xyz, rot_matrix_zyx
from simulator.visualization import GeneralPlotter


class AttitudeView(GeneralPlotter):
    def __init__(self, fig: plt.Figure, pos: int = 111):
        ax = fig.add_subplot(pos, projection="3d")
        super().__init__(fig, ax, pos, is_3d=True)
        self._setup_aircraft_model()

    def _setup_aircraft_model(self):
        # Constants defining the aircraft geometry
        self.SPAN = 14.0
        self.CHORD = 2.0
        self.LENGTH = 10.0
        self.HEIGHT = 1.0
        self.TAIL_HEIGHT = 2.0
        self.TAIL_SPAN = 4.0
        self.TAIL_ROOT = 2.0
        self.TAIL_TIP = 1.0

        # Define subplot for attitude visualization
        self.ax.set_title("Attitude Visualization")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.set_zlabel("Height (m)")
        self.ax.set_xlim(-0.6 * self.SPAN, 0.6 * self.SPAN)
        self.ax.set_ylim(-0.6 * self.SPAN, 0.6 * self.SPAN)
        self.ax.set_zlim(-0.6 * self.SPAN, 0.6 * self.SPAN)

        # Define the vertices of the aircraft components
        self.wing = np.array(
            [
                [-self.CHORD, -0.5 * self.SPAN, 0],
                [0, -0.5 * self.SPAN, 0],
                [0, +0.5 * self.SPAN, 0],
                [-self.CHORD, +0.5 * self.SPAN, 0],
            ]
        )
        self.body = np.array(
            [
                [0.2 * self.LENGTH, 0, 0],
                [0.2 * self.LENGTH, 0, self.HEIGHT],
                [-0.8 * self.LENGTH, 0, self.HEIGHT],
                [-0.8 * self.LENGTH, 0, 0],
            ]
        )
        self.htail = np.array(
            [
                [-0.8 * self.LENGTH + self.TAIL_ROOT, 0, 0],
                [-0.8 * self.LENGTH, 0, 0],
                [-0.8 * self.LENGTH, 0, -self.TAIL_HEIGHT],
                [-0.8 * self.LENGTH + self.TAIL_TIP, 0, -self.TAIL_HEIGHT],
            ]
        )
        self.vtail = np.array(
            [
                [-0.8 * self.LENGTH + self.TAIL_ROOT, 0, 0],
                [-0.8 * self.LENGTH + self.TAIL_TIP, -0.5 * self.TAIL_SPAN, 0],
                [-0.8 * self.LENGTH, -0.5 * self.TAIL_SPAN, 0],
                [-0.8 * self.LENGTH, +0.5 * self.TAIL_SPAN, 0],
                [-0.8 * self.LENGTH + self.TAIL_TIP, +0.5 * self.TAIL_SPAN, 0],
                [-0.8 * self.LENGTH + self.TAIL_ROOT, 0, 0],
            ]
        )

        # Create a collection of the aircraft polygons for attitude visualization
        self.poly = Poly3DCollection(
            [ned2xyz(self.body), ned2xyz(self.wing), ned2xyz(self.htail), ned2xyz(self.vtail)]
        )
        self.poly.set_facecolors(["w", "r", "r", "r"])
        self.poly.set_edgecolor("k")
        self.poly.set_alpha(0.8)
        self.ax.add_collection(self.poly)

    def update(self, state: AircraftState, time: float = None):
        R_vb = state.R_vb
        rotated_body = self.body.dot(R_vb)
        rotated_wing = self.wing.dot(R_vb)
        rotated_htail = self.htail.dot(R_vb)
        rotated_vtail = self.vtail.dot(R_vb)
        self.poly.set_verts(
            [
                ned2xyz(rotated_body),
                ned2xyz(rotated_wing),
                ned2xyz(rotated_htail),
                ned2xyz(rotated_vtail),
            ]
        )
    
