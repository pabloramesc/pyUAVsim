"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.highlighter import ReprHighlighter
from rich.theme import Theme

from simulator.utils.simulation_data import *
from simulator.utils.readable import seconds_to_dhms
from simulator.autopilot.waypoints import Waypoint
from simulator.autopilot.waypoint_actions import (
    OrbitTurns,
    OrbitTime,
    OrbitAlt,
    GoWaypoint,
    SetAirspeed,
)

class SimConsoleLive:
    """
    Handles the display of simulation data using the Rich library without causing flickering by using Live rendering.
    """

    def __init__(self) -> None:
        self.t0 = time.time()
        self.layout = self.create_layout()
        self.theme = Theme({"repr.number": "bold green"})
        self.highlighter = ReprHighlighter()
        self.console = Console(highlighter=self.highlighter, theme=self.theme)
        self.live = Live(self.layout, console=self.console, refresh_per_second=10)
        self.live.start()

    def update(self, data: SimulationData) -> None:
        self.layout["header"].update(self.generate_header(data))
        self.layout["left_top"].update(self.generate_left_top(data))
        self.layout["left_bottom"].update(self.generate_left_bottom(data))
        self.layout["right"].update(self.generate_right(data))
        self.layout["footer1"].update(self.generate_footer1(data))
        self.layout["footer2"].update(self.generate_footer2(data))

    def create_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", size=12),
            Layout(name="footer1", size=7),
            Layout(name="footer2"),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=5),
            Layout(name="right", ratio=4),
        )
        layout["left"].split_column(
            Layout(name="left_top", size=6),
            Layout(name="left_bottom", size=6),
        )
        return layout

    def generate_header(self, data: SimulationData) -> Panel:
        t_real = time.time() - self.t0
        t_sim_str = seconds_to_dhms(data.t_sim)
        t_real_str = seconds_to_dhms(t_real)

        raw_text = (
            f"Real Time: {t_real_str}, "
            f"Sim Time: {t_sim_str}, "
            f"Sim Time Step: {data.dt_sim:.4f} s, "
            f"Sim Iterations: {data.k_sim:d}"
        )
        text = Text.from_markup(raw_text)
        self.highlighter.highlight(text)

        return Panel(
            text,
            title="Simulation Status",
            title_align="left",
            border_style="bold blue",
        )

    def generate_left_top(self, data: SimulationData) -> Panel:
        deltas = data.control_deltas

        table = Table(show_header=False, box=None)
        table.add_column("Delta")
        table.add_column("Value", style="bold green", justify="right")
        table.add_column("Units")

        table.add_row("Aileron:", f"{np.rad2deg(deltas.delta_a):.2f}", "deg")
        table.add_row("Elevator:", f"{np.rad2deg(deltas.delta_e):.2f}", "deg")
        table.add_row("Rudder:", f"{np.rad2deg(deltas.delta_r):.2f}", "deg")
        table.add_row("Throttle:", f"{deltas.delta_t * 100.0:.2f}", "%")

        return Panel(
            table,
            title="Control Deltas",
            title_align="left",
            border_style="bold blue",
        )

    def generate_left_bottom(self, data: SimulationData) -> Panel:
        state = data.uav_state

        # Create a table with 7 columns
        table = Table(show_header=False, box=None)

        table.add_column("Variable")
        table.add_column("X Name", justify="right")
        table.add_column("X Value", style="bold green", justify="right")
        table.add_column("Y Name", justify="right")
        table.add_column("Y Value", style="bold green", justify="right")
        table.add_column("Z Name", justify="right")
        table.add_column("Z Value", style="bold green", justify="right")

        # Add rows for each category
        table.add_row(
            "NED Position (m):",
            "pn:",
            f"{state.pn:.2f}",
            "pe:",
            f"{state.pe:.2f}",
            "pd:",
            f"{state.pd:.2f}",
        )
        table.add_row(
            "Attitude (deg):",
            "roll:",
            f"{np.rad2deg(state.roll):.2f}",
            "pitch:",
            f"{np.rad2deg(state.pitch):.2f}",
            "yaw:",
            f"{np.rad2deg(state.yaw):.2f}",
        )
        table.add_row(
            "Velocity (m/s):",
            "u:",
            f"{state.u:.2f}",
            "v:",
            f"{state.v:.2f}",
            "w:",
            f"{state.w:.2f}",
        )
        table.add_row(
            "Angular Rates (deg/s):",
            "p:",
            f"{np.rad2deg(state.p):.2f}",
            "q:",
            f"{np.rad2deg(state.q):.2f}",
            "r:",
            f"{np.rad2deg(state.r):.2f}",
        )

        # Wrap the table in a panel
        return Panel(
            table,
            title="Aircraft State",
            title_align="left",
            border_style="bold blue",
        )

    def generate_right(self, data: SimulationData) -> Panel:
        status = data.autopilot_status

        table = Table()
        table.add_column("Variable")
        table.add_column("Value", style="bold green", justify="right")
        table.add_column("Target", style="bold green", justify="right")
        table.add_column("Error", style="bold green", justify="right")

        table.add_row(
            "Roll (deg)",
            f"{np.rad2deg(status.roll):.2f}",
            f"{np.rad2deg(status.target_roll):.2f}",
            f"{np.rad2deg(status.roll_error):.2f}",
        )
        table.add_row(
            "Course (deg)",
            f"{np.rad2deg(status.course):.2f}",
            f"{np.rad2deg(status.target_course):.2f}",
            f"{np.rad2deg(status.course_error):.2f}",
        )
        table.add_row(
            "Sideslip Angle (deg)",
            f"{np.rad2deg(status.beta):.2f}",
            f"{np.rad2deg(status.target_beta):.2f}",
            f"{np.rad2deg(status.beta_error):.2f}",
        )
        table.add_row(
            "Pitch (deg)",
            f"{np.rad2deg(status.pitch):.2f}",
            f"{np.rad2deg(status.target_pitch):.2f}",
            f"{np.rad2deg(status.pitch_error):.2f}",
        )
        table.add_row(
            "Altitude (m)",
            f"{status.altitude:.2f}",
            f"{status.target_altitude:.2f}",
            f"{status.altitude_error:.2f}",
        )
        table.add_row(
            "Airspeed (m/s)",
            f"{status.airspeed:.2f}",
            f"{status.target_airspeed:.2f}",
            f"{status.airspeed_error:.2f}",
        )

        return Panel(
            table,
            title="Autopilot Status",
            title_align="left",
            border_style="bold blue",
        )

    def generate_footer1(self, data: SimulationData) -> Panel:
        mc = data.mission_control
        wp = mc.target_waypoint
        am = mc.actions_manager
        rm = mc.route_manager
        pf = mc.path_follower

        raw_text = (
            f"Mission Control Status: {self.format_status(mc.status)}, "
            f"Elapsed Time: {seconds_to_dhms(mc.t)}, "
            f"Wait Orbit: {mc.is_on_wait_orbit}, "
            f"Action Running: {mc.is_action_running}\n"
            f"Target Waypoint ID: {wp.id}, " f"Action Code: {wp.action_code}\n"
            f"Action Manager Status: {self.format_status(am.status)}, "
            f"Active Action: {am.active_action_code.upper()}, "
            f"{am.active_action_status}\n"
            f"Route Manager Status: {self.format_status(rm.status)}, "
            f"Target WP Index: {rm.target_index}, "
            f"Distance to WP: [bold green]{rm.get_distance_to_waypoint(mc.pos_ned):.1f}[/] m\n"
            f"Path Follower Status: {self.format_status(pf.status)}, "
            f"Active Follower: {pf.active_follower_type.upper()}, "
            f"{pf.active_follower_info}, {pf.active_follower_status}\n"
        )
        text = Text.from_markup(raw_text)
        self.highlighter.highlight(text)

        return Panel(
            text,
            title="Mission Status",
            title_align="left",
            border_style="bold blue",
        )

    def generate_footer2(self, data: SimulationData) -> Panel:
        rm = data.mission_control.route_manager

        table = Table()

        table.add_column("Status", style="magenta")
        table.add_column("Index")
        table.add_column("ID")
        table.add_column("PN (m)")
        table.add_column("PE (m)")
        table.add_column("H (m)")
        table.add_column("Action")
        table.add_column("Params")
        table.add_column("Action Status")

        for i, wp in enumerate(rm.waypoints):
            if rm.target_index > i:
                status = "DONE"
                color = "cyan"
            elif rm.target_index == i:
                if rm.status == "end":
                    status = "DONE"
                    color = "cyan"
                elif rm.status == "fail":
                    status = "FAIL"
                    color = "red"
                else:
                    status = ">>>>"
                    color = "green"
            else:
                status = ""
                color = ""

            table.add_row(
                f"[bold]{status}[/bold]",
                f"{i}",
                f"{wp.id}",
                f"{wp.pn:.2f}",
                f"{wp.pe:.2f}",
                f"{wp.h:.2f}",
                f"{wp.action_code.replace('_', ' ')}",
                f"{wp.params}",
                f"{self.action_status(wp)}",
                style=color,
            )

        return Panel(
            table,
            title="Waypoints Table",
            title_align="left",
            border_style="bold blue",
        )

    def format_value(self, value: str) -> str:
        return f"[bold green]{value}[/bold green]"

    def format_status(self, status: str, upper: bool = True) -> str:
        if status.lower() in ["run", "follow"]:
            color = "green"
        elif status.lower() in ["end", "fail"]:
            color = "red"
        elif status.lower() in ["wait", "none"]:
            color = "magenta"
        else:
            color = "cyan"
        _status = status.upper() if upper else status
        return f"[bold {color}]{_status}[/bold {color}]"
        return status

    def action_status(self, wp: Waypoint) -> str:
        if wp.action_code == "ORBIT_TURNS":
            orbit_turns: OrbitTurns = wp.action
            return f"Turns: {orbit_turns.completed_turns:.1f}"

        if wp.action_code == "ORBIT_TIME":
            orbit_time: OrbitTime = wp.action
            return f"Time: {orbit_time.elapsed_time:.2f} s"

        if wp.action_code == "ORBIT_ALT":
            orbit_alt: OrbitAlt = wp.action
            return f"Alt: {orbit_alt.current_altitude} m"

        if wp.action_code == "GO_WAYPOINT":
            go_wp: GoWaypoint = wp.action
            return f"Repeat: {go_wp.repeat_count}"

        if wp.action_code == "SET_AIRSPEED":
            set_va: SetAirspeed = wp.action
            return f"Done: {set_va.is_done()}"
