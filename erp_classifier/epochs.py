# We use a separate Epoch class to accumulate data from the LSL streams until
# the epoch window is complete. This is done as the protocol requires overallping
# epochs, which make a separate book keeping the more feasible approach

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Epoch:
    event: int = 0  # tracking the 'name'/'marker' the event is associated with
    ts_event: float = 0.0
    tmin: float = -0.2
    tmax: float = 1
    nsamples: int = 120
    nchannels: int = 1
    idx: int = 0

    def __post_init__(self):
        self.buffer = np.zeros((self.nsamples, self.nchannels))
        self.buffer_t = np.zeros(self.nsamples)

    def add_data(self, x: np.ndarray, t: np.ndarray):
        assert (
            len(x) <= len(self.buffer) - self.idx
        ), f"Buffer overflow, trying to add {x.shape}, but only {len(self.buffer) - self.idx} entries left"

        assert (
            t[0] > self.buffer_t[-1]
        ), f"Sample times must monotonically increase, got {t[-1]=} but old data is {self.buffer_t[-1]=}"

        assert len(t) == len(x), "Length of data and time must match"

        self.buffer[self.idx : self.idx + len(x)] = x
        self.buffer_t[self.idx : self.idx + len(t)] = t
        self.idx += len(x)


if __name__ == "__main__":
    e = Epoch(1, 123)
