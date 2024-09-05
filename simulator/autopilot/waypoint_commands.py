from simulator.autopilot.mission_control import MissionControl

class WaypointCommand:
    def __init__(self, params):
        self.params = params

    def execute(self, mission_control: MissionControl) -> None:
        """
        Execute the command using the MissionControl instance.

        Parameters
        ----------
        mission_control : MissionControl
            The MissionControl instance to execute the command on.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")