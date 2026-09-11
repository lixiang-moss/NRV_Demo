"""Causal event filters. Input RAW and decoded ROS messages are never modified."""
import ctypes
import numpy as np


class NoiseFilter:
    def __init__(self):
        library = ctypes.CDLL('libnrv_noise_filter.so')
        self.native = library.nrv_filter
        pointer = np.ctypeslib.ndpointer
        self.native.argtypes = [pointer(dtype=np.intp, flags='C_CONTIGUOUS'),
                                pointer(dtype=np.intp, flags='C_CONTIGUOUS'),
                                pointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                pointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                pointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                ctypes.c_bool, ctypes.c_double, ctypes.c_bool, ctypes.c_double,
                                pointer(dtype=np.uint8, flags='C_CONTIGUOUS')]
        self.native.restype = None
        self.shape = None
        self.config = None
        self.last_time = None

    def apply(self, x, y, timestamps, shape, config):
        if self.shape != shape or self.config != config or (
                len(timestamps) and self.last_time is not None and timestamps[0] < self.last_time):
            self.shape, self.config = shape, dict(config)
            self.seen = np.full(shape, -np.inf)
            self.accepted = np.full(shape, -np.inf)
        if len(timestamps):
            self.last_time = timestamps[-1]
        if not config['background'] and not config['refractory']:
            return np.ones(len(x), dtype=bool)
        keep = np.empty(len(x), dtype=np.uint8)
        self.native(x, y, timestamps, len(x), shape[1], shape[0], self.seen, self.accepted,
                    config['background'], config['window_ms'] / 1000.0,
                    config['refractory'], config['interval_ms'] / 1000.0, keep)
        return keep.astype(bool)
