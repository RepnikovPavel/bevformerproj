"""Decord python package"""
from . import bridge, function, logging
from . import ndarray as nd
from ._ffi.base import DECORDError, __version__
from ._ffi.function import (
    extract_ext_funcs,
    get_global_func,
    list_global_func_names,
    register_func,
)
from ._ffi.runtime_ctypes import TypeCode
from .base import ALL
from .ndarray import cpu, gpu
from .video_loader import VideoLoader
from .video_reader import VideoReader

logging.set_level(logging.ERROR)
