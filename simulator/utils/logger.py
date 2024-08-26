import numpy as np


class DataLogger:
    """
    A class for logging time-series data with multiple variables.

    Attributes
    ----------
    nvars : int
        Number of variables being logged.
    labels : list of str
        Labels for each variable.
    buff_size : int
        Size of the buffer.
    count : int
        Total count of logged entries.
    time : np.ndarray
        Array to store time stamps.
    data : np.ndarray
        Array to store variable data.
    """

    def __init__(
        self, nvars: int = 1, labels: list[str] = None, buff_size: int = 100
    ) -> None:
        """Initialize the DataLogger with the specified number of variables, labels, and buffer size.

        Parameters
        ----------
        nvars : int, optional
            The number of variables to log. Default is 1.
        labels : list of str, optional
            A list of labels for each variable. If not provided, default labels will be generated
            in the form ['var1', 'var2', ..., 'varN'] where N is `nvars`.
        buff_size : int, optional
            The initial size of the buffer. This determines the number of data entries the logger
            can store before needing to dynamically expand the buffer. Default is 100.

        Raises
        ------
        ValueError
            If `nvars` is not a positive integer or `buff_size` is not a positive integer.

        """
        if not isinstance(nvars, int) or nvars <= 0:
            raise ValueError("nvars must be a positive integer!")
        self.nvars = nvars

        self.labels = labels if labels else [f"var{k+1}" for k in range(nvars)]

        if not isinstance(buff_size, int) or buff_size <= 0:
            raise ValueError("buff_size must be a positive integer!")
        self.buff_size = buff_size

        self.count = 0
        self.time = np.zeros(buff_size)
        self.data = np.zeros((buff_size, nvars))

    def update(self, t: float, values: np.ndarray) -> None:
        """
        Update the logger with a new timestamp and corresponding variable data.

        Parameters
        ----------
        t : float
            The timestamp for the new data entry.
        values : np.ndarray
            A 1D array containing the new data values for each variable. Must match nvars in length.

        Raises
        ------
        ValueError
            If the size of `values` does not match `nvars`.
        """
        if values.size != self.nvars:
            raise ValueError(f"values must be an array of size {self.nvars}!")

        if self.count >= self.time.size:
            self._extend_buffer()

        self.time[self.count] = t
        self.data[self.count, :] = values
        self.count += 1

    def _extend_buffer(self) -> None:
        """
        Extend the buffer size when capacity is reached.
        """
        self.buff_size *= 2  # Double the buffer size
        self.time = np.resize(self.time, self.buff_size)
        self.data = np.resize(self.data, (self.buff_size, self.nvars))

    def as_array(self) -> np.ndarray:
        """
        Return the logged data as a NumPy array.

        Returns
        -------
        np.ndarray
            A 2D array where the first column is the time data and subsequent columns are the variable data.
        """
        log_array = np.zeros((self.count, self.nvars + 1))
        log_array[:, 0] = self.time[: self.count]
        log_array[:, 1:] = self.data[: self.count, :]
        return log_array

    def as_dict(self) -> dict[str, np.ndarray]:
        """
        Return the logged data as a dictionary.

        Returns
        -------
        dict of str to np.ndarray
            A dictionary where keys are 'time' and the variable labels, and values are the corresponding data arrays.
        """
        log_dict = {"time": self.time[: self.count]}
        for i in range(self.nvars):
            log_dict[self.labels[i]] = self.data[: self.count, i]
        return log_dict
