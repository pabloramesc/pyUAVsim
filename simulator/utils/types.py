from typing import SupportsFloat, Union, Sequence
import numpy as np
from numpy.typing import NDArray

FloatLike = Union[SupportsFloat, np.floating]
FloatArray = Union[NDArray[np.floating], Sequence[FloatLike]]
