"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulator.math.rotation import ned2xyz, rot_matrix_zyx
from simulator.gui.panel_components import View


# Constants defining the aircraft geometry
UAV_SPAN = 14.0
UAV_CHORD = 2.0
UAV_LENGTH = 10.0
UAV_HEIGHT = 1.0
UAV_TAIL_HEIGHT = 2.0
UAV_TAIL_SPAN = 4.0
UAV_TAIL_ROOT = 2.0
UAV_TAIL_TIP = 1.0


class AttitudeView(View):
    """
    A class for creating and managing a 3D attitude visualization of an aircraft model.

    This class extends the View class and provides a 3D representation of an aircraft,
    which can be rotated based on input Euler angles.

    Attributes
    ----------
    wing : np.ndarray
        Vertices defining the wing geometry.
    body : np.ndarray
        Vertices defining the body geometry.
    htail : np.ndarray
        Vertices defining the horizontal tail geometry.
    vtail : np.ndarray
        Vertices defining the vertical tail geometry.
    poly : Poly3DCollection
        A collection of polygons representing the aircraft components.
    """

    def __init__(self, fig: plt.Figure, pos: int = 111):
        super().__init__(fig, pos, is_3d=True)
        self._setup_aircraft_model()

    def _setup_aircraft_model(self):
        """
        Sets up the aircraft model's geometric representation in 3D space.

        This method initializes the vertices for the aircraft's body, wings,
        horizontal tail, and vertical tail, and adds them as polygons to the plot.
        """
        # Define subplot for attitude visualization
        self.ax.set_title("Attitude Visualization")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.set_zlabel("Height (m)")
        self.ax.set_xlim(-0.6 * UAV_SPAN, 0.6 * UAV_SPAN)
        self.ax.set_ylim(-0.6 * UAV_SPAN, 0.6 * UAV_SPAN)
        self.ax.set_zlim(-0.6 * UAV_SPAN, 0.6 * UAV_SPAN)

        # Define the vertices of the aircraft components
        self.wing = np.array(
            [
                [-UAV_CHORD, -0.5 * UAV_SPAN, 0],
                [0, -0.5 * UAV_SPAN, 0],
                [0, +0.5 * UAV_SPAN, 0],
                [-UAV_CHORD, +0.5 * UAV_SPAN, 0],
            ]
        )
        self.body = np.array(
            [
                [0.2 * UAV_LENGTH, 0, 0],
                [0.2 * UAV_LENGTH, 0, UAV_HEIGHT],
                [-0.8 * UAV_LENGTH, 0, UAV_HEIGHT],
                [-0.8 * UAV_LENGTH, 0, 0],
            ]
        )
        self.htail = np.array(
            [
                [-0.8 * UAV_LENGTH + UAV_TAIL_ROOT, 0, 0],
                [-0.8 * UAV_LENGTH, 0, 0],
                [-0.8 * UAV_LENGTH, 0, -UAV_TAIL_HEIGHT],
                [-0.8 * UAV_LENGTH + UAV_TAIL_TIP, 0, -UAV_TAIL_HEIGHT],
            ]
        )
        self.vtail = np.array(
            [
                [-0.8 * UAV_LENGTH + UAV_TAIL_ROOT, 0, 0],
                [-0.8 * UAV_LENGTH + UAV_TAIL_TIP, -0.5 * UAV_TAIL_SPAN, 0],
                [-0.8 * UAV_LENGTH, -0.5 * UAV_TAIL_SPAN, 0],
                [-0.8 * UAV_LENGTH, +0.5 * UAV_TAIL_SPAN, 0],
                [-0.8 * UAV_LENGTH + UAV_TAIL_TIP, +0.5 * UAV_TAIL_SPAN, 0],
                [-0.8 * UAV_LENGTH + UAV_TAIL_ROOT, 0, 0],
            ]
        )

        # Create a collection of the aircraft polygons for attitude visualization
        self.poly = Poly3DCollection(
            [
                ned2xyz(self.body),
                ned2xyz(self.wing),
                ned2xyz(self.htail),
                ned2xyz(self.vtail),
            ]
        )
        self.poly.set_facecolors(["w", "r", "r", "r"])
        self.poly.set_edgecolor("k")
        self.poly.set_alpha(0.8)
        self.ax.add_collection(self.poly)

        self.setup_blit([self.poly])

    def update_view(self, euler: np.ndarray) -> None:
        """
        Updates the aircraft's attitude represenation based on Euler angles.

        The method rotates the aircraft model using the provided Euler angles
        and updates the 3D plot to reflect the new orientation.

        Parameters
        ----------
        euler : np.ndarray
            3-size array with Euler angles [roll, pitch, yaw] in radians.
        """
        R_vb = rot_matrix_zyx(euler)
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

        self.render()
