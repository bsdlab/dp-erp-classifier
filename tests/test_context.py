from pathlib import Path

import yaml

from erp_classifier.context import get_context
from tests.shared import spawn_lsl_marker_stream, spawn_lsl_stream


def test_context_initialization():

    cfg = yaml.safe_load(open(Path("./configs/general_config.yaml").resolve()))
    ctx = get_context()

    # assert names of stream watchers are correct
    assert ctx.input_sw.name == cfg["input"]["lsl_signal_stream_name"]
    assert ctx.classifier_cfg["fband"] == cfg["decoding"]["classifier"]["fband"]


def test_filterbank_setup(spawn_lsl_stream, spawn_lsl_marker_stream):

    ctx = get_context()

    ctx.connect_input(stream_name="test", marker_stream_name="marker")
    assert ctx.filter_bank.ring_buffer.buffer.shape == (
        500,
        5,
        1,
    )  # 5seconds at 100hz, 5 channels, 1 band
    assert list(ctx.filter_bank.bands.keys()) == ["b1"]
    assert list(ctx.filter_bank.bands.values()) == [ctx.classifier_cfg["fband"]]
