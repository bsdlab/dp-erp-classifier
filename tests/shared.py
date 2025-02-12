import threading
import time

import numpy as np
import pylsl
import pytest

from erp_classifier.context import get_context
from erp_classifier.logging import logger


def provide_lsl_stream(
    stop_event: threading.Event, srate: float = 100, nsamples: int = 1000
):
    outlet = pylsl.StreamOutlet(
        pylsl.StreamInfo("test", "test", 5, 100, "float32", "test")
    )
    data = np.tile(np.linspace(0, 1, nsamples), (5, 1))
    data = data.T * np.arange(1, 6)  # 5 channels with linear increase
    data = data.astype(np.float32)

    isampl = 0
    nsent = 0
    tstart = time.perf_counter()
    while not stop_event.is_set():
        dt = time.perf_counter() - tstart
        req_samples = int((dt) * srate) - nsent
        if req_samples > 0:
            outlet.push_chunk(data[isampl : isampl + req_samples, :].tolist())
            nsent += req_samples
            isampl = (isampl + req_samples) % data.shape[0]  # wrap around

        time.sleep(1 / srate)


def provide_lsl_marker_stream(
    stop_event: threading.Event,
):
    outlet = pylsl.StreamOutlet(
        pylsl.StreamInfo(
            "markers", "markers", 1, pylsl.IRREGULAR_RATE, pylsl.cf_int64, "markers"
        )
    )

    markers = [101, 102, 103, 104, 103, 102]

    soa = 0.250
    j = 0

    tstart = time.perf_counter()
    while not stop_event.is_set():
        dt = time.perf_counter() - tstart
        if dt > soa:
            mrk = markers[j % len(markers)]
            j += 1
            logger.debug(f"Sending marker: {mrk=}")
            outlet.push_sample([mrk])
            tstart = time.perf_counter()

        time.sleep(soa / 10)


@pytest.fixture(scope="session")
def spawn_lsl_stream():

    stop_event = threading.Event()
    stop_event.clear()
    th = threading.Thread(target=provide_lsl_stream, args=(stop_event,))
    th.start()

    yield stop_event

    # teardown
    stop_event.set()
    th.join()


@pytest.fixture(scope="session")
def spawn_lsl_marker_stream():

    stop_event = threading.Event()
    stop_event.clear()
    thmrk = threading.Thread(target=provide_lsl_marker_stream, args=(stop_event,))
    thmrk.start()

    yield stop_event

    # teardown
    stop_event.set()
    thmrk.join()


@pytest.fixture
def ctx():
    ctx = get_context()
    return ctx
