"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from rich.console import Console
from rich.table import Table

from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.control_deltas import ControlDeltas
from simulator.autopilot.autopilot_status import AutopilotStatus
from simulator.utils.readable import seconds_to_dhms, seconds_to_hhmmss


class SimConsole:
    """
    A class to handle the display of simulation data using the Rich library.

    Attributes
    ----------
    console : Console
        An instance of the Rich Console class used for printing output.
    """

    def __init__(self) -> None:
        """
        Initializes the SimConsole with a Rich Console instance.
        """
        self.console = Console()

    def print_time(
        self,
        t_sim: float,
        t_real: float,
        dt_sim: float = None,
        k_sim: int = None,
        style="simple",
    ) -> None:
        """
        Prints the simulation and real time, along with the time step and iteration count if provided.

        Parameters
        ----------
        t_sim : float
            The current simulation time in seconds.
        t_real : float
            The current real time in seconds.
        dt_sim : float, optional
            The time step of the simulation in seconds (default is None).
        k_sim : int, optional
            The current iteration count of the simulation (default is None).
        style : str, optional
            The style of output, either 'simple' or 'table' (default is 'simple').

        Raises
        ------
        ValueError
            If the style is not 'simple' or 'table'.
        """
        if style == "simple":
            self._print_time_simple(t_sim, t_real, dt_sim, k_sim)
        elif style == "table":
            self._print_time_table(t_sim, t_real, dt_sim, k_sim)
        else:
            raise ValueError(
                "Not valid style parameter! Valid options are 'simple' or 'table'."
            )

    def print_control_deltas(self, deltas: ControlDeltas, style="simple") -> None:
        """
        Prints the control deltas for the aircraft.

        Parameters
        ----------
        deltas : ControlDeltas
            An instance of the ControlDeltas class containing control inputs.
        style : str, optional
            The style of output, either 'simple' or 'table' (default is 'simple').

        Raises
        ------
        ValueError
            If the style is not 'simple' or 'table'.
        """
        if style == "simple":
            self._print_control_deltas_simple(deltas)
        elif style == "table":
            self._print_control_deltas_table(deltas)
        else:
            raise ValueError(
                "Not valid style parameter! Valid options are 'simple' or 'table'."
            )

    def print_aircraft_state(self, state: AircraftState, style="simple") -> None:
        """
        Prints the state of the aircraft in the specified style.

        Parameters
        ----------
        state : AircraftState
            An instance of the AircraftState class representing the current state of the aircraft.
        style : str, optional
            The style of output, either 'simple' or 'table' (default is 'simple').

        Raises
        ------
        ValueError
            If the style is not 'simple' or 'table'.
        """
        if style == "simple":
            self._print_aircraft_state_simple(state)
        elif style == "table":
            self._print_aircraft_state_table(state)
        else:
            raise ValueError(
                "Not valid style parameter! Valid options are 'simple' or 'table'."
            )

    def print_autopilot_status(self, status: AutopilotStatus, style="simple") -> None:
        """
        Prints the current status of the autopilot.

        Parameters
        ----------
        status : AutopilotStatus
            An instance of the AutopilotStatus class representing the current autopilot status.
        style : str, optional
            The style of output, either 'simple' or 'table' (default is 'simple').

        Raises
        ------
        ValueError
            If the style is not 'simple' or 'table'.
        """
        if style == "simple":
            self._print_autopilot_status_simple(status)
        elif style == "table":
            self._print_autopilot_status_table(status)
        else:
            raise ValueError(
                "Not valid style parameter! Valid options are 'simple' or 'table'."
            )

    def _print_time_simple(
        self, t_sim: float, t_real: float, dt_sim: float = None, k_sim: int = None
    ) -> None:
        """
        Prints the simulation and real time in a simple text format.

        Parameters
        ----------
        t_sim : float
            The current simulation time in seconds.
        t_real : float
            The current real time in seconds.
        dt_sim : float, optional
            The time step of the simulation in seconds (default is None).
        k_sim : int, optional
            The current iteration count of the simulation (default is None).
        """
        self.console.clear()
        t_sim_str = seconds_to_dhms(t_sim)
        t_real_str = seconds_to_dhms(t_real)
        txt = f"Real Time: {t_real_str}, Sim Time: {t_sim_str}"
        if dt_sim is not None:
            txt += f", Sim Time Step: {dt_sim:.4f} s"
        if k_sim is not None:
            txt += f", Sim Iterations: {k_sim:d}"
        self.console.print(txt)
        self.console.rule()

    def _print_time_table(
        self, t_sim: float, t_real: float, dt_sim: float = None, k_sim: int = None
    ) -> None:
        """
        Prints the simulation and real time in a tabular format.

        Parameters
        ----------
        t_sim : float
            The current simulation time in seconds.
        t_real : float
            The current real time in seconds.
        dt_sim : float, optional
            The time step of the simulation in seconds (default is None).
        k_sim : int, optional
            The current iteration count of the simulation (default is None).
        """
        self.console.clear()
        table = Table()

        # Define columns
        table.add_column("Real Time", style="magenta")
        table.add_column("Sim Time", style="cyan")
        table.add_column("Sim Time Step", style="cyan")
        table.add_column("Sim Iterations", style="cyan")

        # Add a single row with values
        row = [seconds_to_hhmmss(t_real), seconds_to_hhmmss(t_sim)]
        if dt_sim is not None:
            row.append(f"{dt_sim:.4f} s")
        else:
            row.append("N/A")

        if k_sim is not None:
            row.append(f"{k_sim:d}")
        else:
            row.append("N/A")

        table.add_row(*row)

        self.console.print(table)

    def _print_control_deltas_simple(self, deltas: ControlDeltas) -> None:
        """
        Prints the control deltas in a simple text format.

        Parameters
        ----------
        deltas : ControlDeltas
            An instance of the ControlDeltas class containing control inputs.
        """
        self.console.print(
            "[bold magenta underline]Control Deltas[/bold magenta underline]"
        )
        self.console.print(f"Aileron: {np.rad2deg(deltas.delta_a):.2f} deg")
        self.console.print(f"Elevator: {np.rad2deg(deltas.delta_e):.2f} deg")
        self.console.print(f"Rudder: {np.rad2deg(deltas.delta_r):.2f} deg")
        self.console.print(f"Throttle: {deltas.delta_t * 100.0:.2f} %")
        self.console.rule()

    def _print_control_deltas_table(self, deltas: ControlDeltas) -> None:
        """
        Prints the control deltas in a tabular format.

        Parameters
        ----------
        deltas : ControlDeltas
            An instance of the ControlDeltas class containing control inputs.
        """
        table = Table()

        # Define columns
        table.add_column("Control Deltas", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add rows to the table
        table.add_row("Aileron", f"{np.rad2deg(deltas.delta_a):.2f} deg")
        table.add_row("Elevator", f"{np.rad2deg(deltas.delta_e):.2f} deg")
        table.add_row("Rudder", f"{np.rad2deg(deltas.delta_r):.2f} deg")
        table.add_row("Throttle", f"{deltas.delta_t * 100.0:.2f} %")

        self.console.print(table)

    def _print_aircraft_state_simple(self, state: AircraftState) -> None:
        """
        Prints the aircraft state in a simple text format.

        Parameters
        ----------
        state : AircraftState
            An instance of the AircraftState class representing the current state of the aircraft.
        """
        self.console.print(
            "[bold magenta underline]Aircraft State[/bold magenta underline]"
        )
        self.console.print(
            f"NED position (m): pn: {state.pn:.2f}, pe: {state.pe:.2f}, pd: {state.pd:.2f}"
        )
        self.console.print(
            f"Body velocity (m/s): u: {state.u:.2f}, v: {state.v:.2f}, w: {state.w:.2f}"
        )
        self.console.print(
            f"Attitude (degs): roll: {np.rad2deg(state.roll):.2f}, pitch: {np.rad2deg(state.pitch):.2f}, yaw: {np.rad2deg(state.yaw):.2f}"
        )
        self.console.print(
            f"Angular Rates (rads/s): p: {state.p:.2f}, q: {state.q:.2f}, r: {state.r:.2f}"
        )
        self.console.rule()

    def _print_aircraft_state_table(self, state: AircraftState) -> None:
        """
        Prints the aircraft state in a tabular format.

        Parameters
        ----------
        state : AircraftState
            An instance of the AircraftState class representing the current state of the aircraft.
        """
        table = Table()

        # Define columns
        table.add_column("Aircraft State", justify="left", style="cyan", no_wrap=True)
        table.add_column("X Value", style="magenta")
        table.add_column("Y Value", style="green")
        table.add_column("Z Value", style="blue")

        # Add rows to the table
        table.add_row(
            "NED Position (m)",
            f"pn: {state.pn:.2f}",
            f"pe: {state.pe:.2f}",
            f"pd: {state.pd:.2f}",
        )
        table.add_row(
            "Body Velocity (m/s)",
            f"u: {state.u:.2f}",
            f"v: {state.v:.2f}",
            f"w: {state.w:.2f}",
        )
        table.add_row(
            "Attitude (deg)",
            f"roll: {np.rad2deg(state.roll):.2f}",
            f"pitch: {np.rad2deg(state.pitch):.2f}",
            f"yaw: {np.rad2deg(state.yaw):.2f}",
        )
        table.add_row(
            "Angular Rates (rad/s)",
            f"p: {state.p:.2f}",
            f"q: {state.q:.2f}",
            f"r: {state.r:.2f}",
        )

        self.console.print(table)

    def _print_autopilot_status_simple(self, status: AutopilotStatus) -> None:
        """
        Prints the autopilot status in a simple text format.

        Parameters
        ----------
        status : AutopilotStatus
            An instance of the AutopilotStatus class representing the current autopilot status.
        """
        self.console.print(
            "[bold magenta underline]Autopilot Status[/bold magenta underline]"
        )
        self.console.print(
            f"Roll: {np.rad2deg(status.roll):.2f} deg, Target Roll: {np.rad2deg(status.target_roll):.2f} deg, Error: {np.rad2deg(status.roll_error):.2f} deg"
        )
        self.console.print(
            f"Course: {np.rad2deg(status.course):.2f} deg, Target Course: {np.rad2deg(status.target_course):.2f} deg, Error: {np.rad2deg(status.course_error):.2f} deg"
        )
        self.console.print(
            f"Sideslip Angle: {np.rad2deg(status.beta):.2f} deg, Target Sideslip Angle: {np.rad2deg(status.target_beta):.2f} deg, Error: {np.rad2deg(status.beta_error):.2f} deg"
        )
        self.console.print(
            f"Pitch: {np.rad2deg(status.pitch):.2f} deg, Target Pitch: {np.rad2deg(status.target_pitch):.2f} deg, Error: {np.rad2deg(status.pitch_error):.2f} deg"
        )
        self.console.print(
            f"Altitude: {status.altitude:.2f} m, Target Altitude: {status.target_altitude:.2f} m, Error: {status.altitude_error:.2f} m"
        )
        self.console.print(
            f"Airspeed: {status.airspeed:.2f} m/s, Target Airspeed: {status.target_airspeed:.2f} m/s, Error: {status.airspeed_error:.2f} m/s"
        )
        self.console.rule()

    def _print_autopilot_status_table(self, status: AutopilotStatus) -> None:
        """
        Prints the autopilot status in a tabular format.

        Parameters
        ----------
        status : AutopilotStatus
            An instance of the AutopilotStatus class representing the current autopilot status.
        """
        table = Table()

        # Define columns
        table.add_column("Autopilot Status", justify="left", style="cyan", no_wrap=True)
        table.add_column("Current Value", style="magenta")
        table.add_column("Target Value", style="green")
        table.add_column("Error", style="red")

        # Add rows to the table
        table.add_row(
            "Roll",
            f"{np.rad2deg(status.roll):.2f} deg",
            f"{np.rad2deg(status.roll_target):.2f} deg",
            f"{np.rad2deg(status.roll_error):.2f} deg",
        )
        table.add_row(
            "Course",
            f"{np.rad2deg(status.course):.2f} deg",
            f"{np.rad2deg(status.course_target):.2f} deg",
            f"{np.rad2deg(status.course_error):.2f} deg",
        )
        table.add_row(
            "Sideslip Angle",
            f"{np.rad2deg(status.beta):.2f} deg",
            f"{np.rad2deg(status.beta_target):.2f} deg",
            f"{np.rad2deg(status.beta_error):.2f} deg",
        )
        table.add_row(
            "Pitch",
            f"{np.rad2deg(status.pitch):.2f} deg",
            f"{np.rad2deg(status.pitch_target):.2f} deg",
            f"{np.rad2deg(status.pitch_error):.2f} deg",
        )
        table.add_row(
            "Altitude",
            f"{status.altitude:.2f} m",
            f"{status.altitude_target:.2f} m",
            f"{status.altitude_error:.2f} m",
        )
        table.add_row(
            "Airspeed",
            f"{status.airspeed:.2f} m/s",
            f"{status.airspeed_target:.2f} m/s",
            f"{status.airspeed_error:.2f} m/s",
        )

        self.console.print(table)
