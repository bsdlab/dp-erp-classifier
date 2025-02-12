from dataclasses import dataclass, field
from pathlib import Path

import pylsl
import yaml
from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher

# TODO: Probably we should directly use Jan's library to reduce dependencies
from pyclf.lda.classification import EpochsVectorizer
from sklearn.pipeline import Pipeline


class CtxBase:
    # nicer aligment of the __repr__ with args in different lines each
    def __repr__(self):
        return (
            self.__class__.__name__
            + "(\n  "
            + ",\n  ".join(
                [
                    f"{k}={v}" if not isinstance(v, str) else f'{k}="{v}"'
                    for k, v in self.__dict__.items()
                ]
            )
            + "\n)"
        )


@dataclass
class ClassifierContext(CtxBase):

    dt_s: float = 0.1
    input_signal_stream_name: str = ""
    input_marker_stream_name: str = ""
    output_stream_name: str = ""
    decode_trigger_markers: list = field(default_factory=lambda: [101, 102])
    input_sw: StreamWatcher = field(init=False)
    input_mrk_sw: StreamWatcher = field(init=False)
    filter_bank: FilterBank = field(init=False)
    classifier_cfg: dict = field(default_factory=dict)
    vectorizer: EpochsVectorizer = field(init=False)
    epo_tmax_s: float = 1
    clf_pipeline: Pipeline = field(init=False)
    outlet: pylsl.StreamOutlet = field(init=False)
    epochs_accumulating_stack: list = field(default_factory=list)
    epochs_for_clf_stack: list = field(default_factory=list)

    # A list of epoch starts containing tuples of (event_id, filter_buffer_time_idx) for currently
    # accumulating epochs. This is used for bookkeeping as we accumulate overlapping
    # epochs. The `filter_buffer_time_idx` is used to get the closest matching time
    # stamp in the filter_bank's buffer (used on the unfolded data).
    current_epos_start: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self):
        self.input_sw = StreamWatcher(self.input_signal_stream_name)
        self.input_mrk_sw = StreamWatcher(self.input_marker_stream_name)

        self.epo_tmax_s = self.classifier_cfg["tmax"]

        self.init_outlet()

    def connect_input(
        self, stream_name: str | None = None, marker_stream_name: str | None = None
    ):
        # if None, the sw.name is used internally
        id_dict = {"name": stream_name} if stream_name else None
        id_dict_mrk = {"name": marker_stream_name} if marker_stream_name else None
        self.input_sw.connect_to_stream(identifier=id_dict)
        self.input_mrk_sw.connect_to_stream(identifier=id_dict_mrk)

        self.filter_bank = FilterBank(
            bands={"b1": self.classifier_cfg["fband"]},
            sfreq=self.input_sw.inlet.info().nominal_srate(),
            n_in_channels=len(self.input_sw.channel_names),
            filter_buffer_s=5,
        )

        # set here as now we know the sampling freq of incoming data (-> no resampling needed as we extract features anyways)
        self.vectorizer = EpochsVectorizer(
            jumping_mean_ivals=self.classifier_cfg["ivals"],
            sfreq=self.input_sw.inlet.info().nominal_srate(),
            t_ref=self.classifier_cfg["tmin"],
        )

    def init_outlet(self):
        """
        Init the outlet as regular string marker, as it is not time critical
        when exactly the classifier result is sent out.
        """

        info = pylsl.StreamInfo(
            name=self.output_stream_name,
            type="Markers",
            channel_format=pylsl.cf_string,
            source_id=self.output_stream_name,
        )
        self.outlet = pylsl.StreamOutlet(info)

    def set_classifier(self, clf_pipeline: Pipeline):
        self.clf_pipeline = clf_pipeline

    def __repr__(self):
        return super().__repr__()

    def update_stream_watchers(self):
        """Update the StreamWatchers and check for new data"""
        self.input_sw.update()
        self.input_mrk_sw.update()

        # If new data -> filter it
        if self.input_sw.n_new > 0:
            x = self.input_sw.unfold_buffer()[-self.input_sw.n_new :]
            ts = self.input_sw.unfold_buffer_t()[-self.input_sw.n_new :]
            self.input_sw.n_new = 0
            self.filter_bank.filter(x, ts)


def get_context() -> ClassifierContext:
    """Load a ClassifierContext based on the config in ./configs/general_config.yaml"""

    cfg = yaml.safe_load(open(Path("./configs/general_config.yaml").resolve()))
    ctx = ClassifierContext(
        input_signal_stream_name=cfg["input"]["lsl_signal_stream_name"],
        input_marker_stream_name=cfg["input"]["lsl_marker_stream_name"],
        output_stream_name=cfg["output"]["lsl_stream_name"],
        dt_s=cfg["decoding"]["processing_loop_dt_s"],
        classifier_cfg=cfg["decoding"]["classifier"],
        decode_trigger_markers=cfg["decoding"]["decode_trigger_markers"],
    )

    return ctx


if __name__ == "__main__":
    ctx = get_context()
