""" Time or Past windowed multimodel variants """

from . import _mm_time, _mm_past
from ._mm_time import *
from ._mm_past import *

__all__ = _mm_time.__all__.copy()
__all__ += _mm_past.__all__.copy()
