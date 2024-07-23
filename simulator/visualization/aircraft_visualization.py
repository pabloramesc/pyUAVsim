import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from simulator.math.rotation import rot_matrix_zyx, ned2xyz

class AircraftVisualization:
    def __init__(self):
        # Constants defining the aircraft geometry
        self.SPAN = 14.0
        self.CHORD = 2.0
        self.LENGTH = 10.0
        self.HEIGHT = 1.0
        self.TAIL_HEIGHT = 2.0
        self.TAIL_SPAN = 4.0
        self.TAIL_ROOT = 2.0
        self.TAIL_TIP = 1.0

        # Enable interactive mode
        plt.ion()

        # Create a figure with a specific layout
        self.fig = plt.figure(figsize=(12, 6))

        # Define subplot for attitude visualization
        self.ax_attitude = self.fig.add_subplot(121, projection="3d")
        self.ax_attitude.set_title("Attitude Visualization")
        self.ax_attitude.set_xlabel("East (m)")
        self.ax_attitude.set_ylabel("North (m)")
        self.ax_attitude.set_zlabel("Height (m)")
        self.ax_attitude.set_xlim(-0.6 * self.SPAN, 0.6 * self.SPAN)
        self.ax_attitude.set_ylim(-0.6 * self.SPAN, 0.6 * self.SPAN)
        self.ax_attitude.set_zlim(-0.6 * self.SPAN, 0.6 * self.SPAN)

        # Define subplot for position visualization
        self.ax_position = self.fig.add_subplot(122, projection="3d")
        self.ax_position.set_title("Position in NED Frame")
        self.ax_position.set_xlabel("East (m)")
        self.ax_position.set_ylabel("North (m)")
        self.ax_position.set_zlabel("Height (m)")
        self.ax_position.set_xlim(-50, 50)
        self.ax_position.set_ylim(-50, 50)
        self.ax_position.set_zlim(-50, 50)

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
        self.ax_attitude.add_collection(self.poly)

        # Initialize empty lists for position lines
        self.line, = self.ax_position.plot([], [], [], color='b', linewidth=2.0)
        
        # Initialize the aircraft's starting position
        self.position_history = []

        # Show the plots
        plt.show()

    def update(self, state: np.ndarray, pause: float = 0.0):
        # Update the angles and position from the state array
        roll, pitch, yaw = state[6:9]
        north, east, down = state[0:3]

        # Rotate the components for attitude visualization
        R_vb = rot_matrix_zyx(np.array([roll, pitch, yaw]))
        rotated_body = self.body.dot(R_vb)
        rotated_wing = self.wing.dot(R_vb)
        rotated_htail = self.htail.dot(R_vb)
        rotated_vtail = self.vtail.dot(R_vb)

        # Update the vertices for attitude visualization
        self.poly.set_verts([
            ned2xyz(rotated_body),
            ned2xyz(rotated_wing),
            ned2xyz(rotated_htail),
            ned2xyz(rotated_vtail)
        ])

        # Update the position history
        self.position_history.append(ned2xyz(np.array([north, east, down])))

        # Update the position visualization
        if len(self.position_history) > 1:
            positions = np.array(self.position_history)
            self.line.set_data(positions[:, 0], positions[:, 1])
            self.line.set_3d_properties(positions[:, 2])

        # Update the plots
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(pause)
