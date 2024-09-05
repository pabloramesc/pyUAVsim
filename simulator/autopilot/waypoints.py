class Waypoint:
    def __init__(
        self, id: int, pn: float, pe: float, h: float, command: str = None, *params
    ):
        self.id = id
        self.pn = pn
        self.pe = pe
        self.h = h
        self.command = command
        self.params = params

    def __repr__(self):
        return (
            f"Waypoint(id={self.id}, pn={self.pn}, pe={self.pe}, h={self.h}, "
            f"command={self.command}, params={self.params})"
        )


class WaypointsList:
    def __init__(self):
        self.waypoints = []

    def add_waypoint(self, waypoint: Waypoint):
        self.waypoints.append(waypoint)

    def get_waypoints(self):
        return self.waypoints

    def load_from_txt(self, filename: str) -> None:
        """
        Load waypoints from a text file and populate the WaypointsList.

        Parameters
        ----------
        filename : str
            The path to the text file containing waypoints data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file content is not in the expected format.
        """
        try:
            with open(filename, "r") as file:
                lines = file.readlines()

            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    raise ValueError(f"Invalid waypoint format: {line}")

                id = int(parts[0].strip())
                pn = float(parts[1].strip())
                pe = float(parts[2].strip())
                h = float(parts[3].strip())

                command = parts[4].strip() if len(parts) > 4 else None
                params = [float(p.strip()) for p in parts[5:]] if len(parts) > 5 else []

                waypoint = Waypoint(id, pn, pe, h, command, *params)
                self.add_waypoint(waypoint)

        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            
        except ValueError as e:
            print(f"Error loading waypoints from TXT: {e}")
