import copy
from pathlib import Path

import mne

# TODO: Consider downloading directly from Freidocs -> as moabb requirements are rather limiting on numpy, pandas, pytest and seaborn...
import moabb
import numpy as np
import pytest
from dareplane_utils.logging.logger import get_logger
from sklearn.metrics import accuracy_score

from erp_classifier.classifier.toeplitz_lda import get_toeplitz_LDA_pipeline
from tests.shared import ctx, spawn_lsl_marker_stream, spawn_lsl_stream

TEST_ASSETS_DIR = Path("./tests/assets")
TEST_ASSETS_DIR.mkdir(exist_ok=True)
MOABB_DIR = TEST_ASSETS_DIR / "moabb_data"
moabb.utils.set_download_dir(MOABB_DIR)

logger = get_logger(
    "erp_classifier_tests", add_console_handler=True, no_socket_handler=True
)
logger.setLevel("DEBUG")


@pytest.fixture(scope="session")
def test_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Use moabb data for testing, we convert the data to numpy arrays

    """

    d = moabb.datasets.Sosulski2019().get_data(
        subjects=[3]
    )  # just picking one, no particular reason

    # get only the 600ms SOA
    d = {k: v for k, v in d[3].items() if "600" in k}
    epos = []
    for k, v in d.items():
        for k0, raw in v.items():
            # fband as used in https://diglib.tugraz.at/download.php?id=5d8090c22a083&location=medra
            raw.filter(1.5, 40)

            ev, evid = mne.events_from_annotations(raw)
            sev_id = {k: v for k, v in evid.items() if k in ["NonTarget", "Target"]}
            epo = mne.Epochs(
                raw,
                ev,
                event_id=sev_id,
                tmin=-0.2,
                tmax=1.2,
                preload=True,
            )
            epo.resample(sfreq=100)
            epos.append(epo.copy())

    # filter all
    epo = mne.concatenate_epochs(epos)
    data = epo.pick_types(eeg=True).get_data(copy=True)
    y = np.asarray(
        [0 if ev[-1] == epo.event_id["NonTarget"] else 1 for ev in epo.events]
    ).reshape(-1, 1)

    return data, y


def test_train_classifier(test_data):
    X, y = test_data
    pl = get_toeplitz_LDA_pipeline(n_channels=X.shape[1])

    with pytest.raises(AttributeError):
        _ = pl.steps[0][-1].classes_

    # for the first test we just connect the channel and time dimension
    Xl = X.reshape(X.shape[0], -1)
    pl.fit(
        Xl, y.ravel()
    )  # Note: while we adhered to the standard dimensions for the labels as used from sklearn, the pyclf library expects a 1D array for the labels

    assert (pl.steps[0][-1].classes_ == np.array([0, 1])).all()


def test_vectorization_of_epochs(
    ctx, test_data, spawn_lsl_stream, spawn_lsl_marker_stream
):
    X, y = test_data

    # connect to an LSL inlet to initialize the vectorizer with correct sfreq
    ctx.connect_input(stream_name="test", marker_stream_name="markers")

    features = ctx.vectorizer.transform(X)
    assert features.shape[0] == X.shape[0]
    assert (
        features.shape[-1] == len(ctx.classifier_cfg["ivals"]) * X.shape[1]
    )  # one features for each time window defined x n_channels


def test_prediction_accuracy(ctx, test_data, spawn_lsl_stream, spawn_lsl_marker_stream):
    X, y = test_data

    # connect to an LSL inlet to initialize the vectorizer with correct sfreq
    ctx.connect_input(stream_name="test", marker_stream_name="markers")

    features = ctx.vectorizer.transform(X)

    pl = get_toeplitz_LDA_pipeline(n_channels=X.shape[1])

    idx_cut_off = int(X.shape[0] * 0.8)

    ftrain = features[:idx_cut_off]
    ftest = features[idx_cut_off:]
    ytrain = y[:idx_cut_off].ravel()
    ytest = y[idx_cut_off:].ravel()

    pl.fit(ftrain, ytrain)

    pred = pl.predict(ftest)
    acc = accuracy_score(ytest, pred)

    logger.debug(f"Achieved {acc:.2%} accuracy_score")

    assert acc > 0.8  # with the initial cfg, it should be ~0.83
