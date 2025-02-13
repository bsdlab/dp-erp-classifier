from pathlib import Path

import yaml

from tests.shared import ctx, spawn_lsl_marker_stream, spawn_lsl_stream


def test_context_initialization(ctx):

    cfg = yaml.safe_load(open(Path("./configs/general_config.yaml").resolve()))

    # assert names of stream watchers are correct
    assert ctx.input_sw.name == cfg["input"]["lsl_signal_stream_name"]
    assert ctx.classifier_cfg["fband"] == cfg["decoding"]["classifier"]["fband"]


def test_filterbank_setup(ctx, spawn_lsl_stream, spawn_lsl_marker_stream):

    ctx.connect_input(stream_name="test", marker_stream_name="markers")
    assert ctx.filter_bank.ring_buffer.buffer.shape == (
        500,
        5,
        1,
    )  # 5seconds at 100hz, 5 channels, 1 band
    assert list(ctx.filter_bank.bands.keys()) == ["b1"]
    assert list(ctx.filter_bank.bands.values()) == [ctx.classifier_cfg["fband"]]


def test_feature_vector_store(ctx, spawn_lsl_stream, spawn_lsl_marker_stream):
    ctx.connect_input(stream_name="test", marker_stream_name="markers")

    assert ctx.feature_vecs.buffer.shape == (
        ctx.feature_vecs_buffer_size,
        len(ctx.input_sw.channel_names) * len(ctx.classifier_cfg["ivals"]),
    )  # 100 samples, 50 features (if 10 ivals)
