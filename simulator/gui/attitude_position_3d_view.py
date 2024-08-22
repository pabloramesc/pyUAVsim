import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from simulator.math.rotation import rot_matrix_zyx, ned2xyz
from simulator.aircraft import AircraftState

class AttitudePosition3DView:
    def __init__(self, figsize=(12, 6), use_blit: bool = False):
        self.use_blit = use_blit

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
        self.fig = plt.figure(figsize=figsize)

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

        # Create a blit background for the attitude and position plots
        if self.use_blit:
            self.attitude_background = self.fig.canvas.copy_from_bbox(self.ax_attitude.bbox)
            self.position_background = self.fig.canvas.copy_from_bbox(self.ax_position.bbox)

        # Show the plots
        plt.show()

    def update(self, state: AircraftState, pause: float = 0.0):
        # Rotate the components for attitude visualization
        rotated_body = self.body.dot(state.R_vb)
        rotated_wing = self.wing.dot(state.R_vb)
        rotated_htail = self.htail.dot(state.R_vb)
        rotated_vtail = self.vtail.dot(state.R_vb)

        # Update the vertices for attitude visualization
        self.poly.set_verts([
            ned2xyz(rotated_body),
            ned2xyz(rotated_wing),
            ned2xyz(rotated_htail),
            ned2xyz(rotated_vtail)
        ])

        # Update the position history
        self.position_history.append(ned2xyz(state.ned_position))

        # Update the position visualization
        if len(self.position_history) > 1:
            positions = np.array(self.position_history)
            self.line.set_data(positions[:, 0], positions[:, 1])
            self.line.set_3d_properties(positions[:, 2])
            # Autoscale axis limits
            self.ax_position.set_xlim(min(positions[:, 0]), max(positions[:, 0]))
            self.ax_position.set_ylim(min(positions[:, 1]), max(positions[:, 1]))
            self.ax_position.set_zlim(min(positions[:, 2]), max(positions[:, 2]))
            
        # Update the plots
        if self.use_blit:
            # Use blit to update only the changed parts of the canvas
            self.fig.canvas.restore_region(self.attitude_background)
            self.ax_attitude.draw_artist(self.poly)
            self.fig.canvas.restore_region(self.position_background)
            self.ax_position.draw_artist(self.line)
            self.fig.canvas.blit(self.ax_attitude.bbox)
            self.fig.canvas.blit(self.ax_position.bbox)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        plt.pause(pause)
