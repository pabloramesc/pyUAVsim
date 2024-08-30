"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np


class LowPassFilter:
    def __init__(self, gain: np.ndarray, init_value: np.ndarray = None) -> None:
        """
        Low Pass Filter

        Parameters
        ----------
        gain : np.ndarray
            filter gain. If 0.0 any filtering will be applied. If 1.0 value wont be updated.
        init_value : np.ndarray, optional
            initial value of the filter, by default None

        Raises
        ------
        TypeError
            If gain or init_value types are not float or numpy.ndarray
        ValueError
            If gain is not between 0.0 and 1.0
        """
        # type cheking
        if not (isinstance(gain, np.ndarray) or isinstance(gain, float)):
            raise TypeError("gain must be a float or a numpy.ndarray!")
        if not (isinstance(init_value, np.ndarray) or isinstance(init_value, float)):
            raise TypeError("init_value must be a float or a numpy.ndarray!")
        if isinstance(gain, float) and (gain > 1.0 or gain < 0.0):
            raise ValueError("gain must be a value between 0.0 and 1.0!")
        if isinstance(gain, np.ndarray) and np.any(gain > 1.0 or gain < 0.0):
            raise ValueError("gain must be a value between 0.0 and 1.0!")
        # initialization
        self.gain = gain
        self.gain_1 = 1.0 - gain
        self.last_value = init_value

    def update(self, value: np.ndarray) -> np.ndarray:
        # type checking
        # if not isinstance(value, np.ndarray):
        #     raise TypeError("value must be a float or a numpy.ndarray!")
        # compute filter next value
        self.last_value = self.gain * self.last_value + self.gain_1 * value
        return self.last_value

    def reset(self, value: np.ndarray) -> np.ndarray:
        self.last_value = value
        return self.last_value


class LowPassFilter_2ndOrder:
    def __init__(self, gain: np.ndarray, init_value: np.ndarray = None) -> None:
        """
        Low Pass Filter

        Parameters
        ----------
        gains : np.ndarray
            filter gain. If 0.0 any filtering will be applied. If 1.0 value wont be updated.
        init_value : np.ndarray, optional
            initial value of the filter, by default None

        Raises
        ------
        TypeError
            If gain or init_value types are not float or numpy.ndarray
        ValueError
            If gain is not between 0.0 and 1.0
        """
        # type cheking
        if not (isinstance(gain, np.ndarray) or isinstance(gain, float)):
            raise TypeError("gain must be a float or a numpy.ndarray!")
        if not (isinstance(init_value, np.ndarray) or isinstance(init_value, float)):
            raise TypeError("init_value must be a float or a numpy.ndarray!")
        if isinstance(gain, float) and (gain > 1.0 or gain < 0.0):
            raise ValueError("gain must be a value between 0.0 and 1.0!")
        if isinstance(gain, np.ndarray) and np.any(gain > 1.0 or gain < 0.0):
            raise ValueError("gain must be a value between 0.0 and 1.0!")
        # initialization
        self.a = gain
        self.b = (1.0 - gain)/2
        self.x_1 = init_value
        self.x_2 = init_value
        self.y_1 = init_value

    def update(self, value: np.ndarray) -> np.ndarray:
        # type checking
        # if not isinstance(value, np.ndarray):
        #     raise TypeError("value must be a float or a numpy.ndarray!")
        # compute filter next value
        self.x_2 = self.x_1
        self.x_1 = value
        self.y_1 = self.a * self.y_1 + self.b * (self.x_1 + self.x_2)
        return self.y_1

    def reset(self, value: np.ndarray) -> np.ndarray:
        self.x_1 = value
        self.x_2 = value
        self.y_1 = value
        return self.y_1
