import numpy as np

from erp_classifier.context import ClassifierContext
from erp_classifier.logging import logger


# TODO: MD consider refactoring how we deal with this. I do not like the current appoach, but will use if for a first iteration
def new_epoch_started(ctx: ClassifierContext) -> tuple[int, int] | None:
    n_new = ctx.input_mrk_sw.n_new
    if n_new == 0:
        return None

    markers = ctx.input_mrk_sw.unfold_buffer()[-n_new:, 0]
    markers_t = ctx.input_mrk_sw.unfold_buffer_t()[-n_new:]
    trigger_marker_indices = [
        i for i, m in enumerate(markers) if m in ctx.decode_trigger_markers
    ]

    if len(trigger_marker_indices) > 1:
        logger.warning(
            f"More than one trigger marker found in the last {n_new=} markers."
            " Only the last one will be processed. Consider increasing the "
            f"refresh rate for the main loop - currently: {ctx.dt_s=} "
        )

    if any(trigger_marker_indices):

        # find the closest match of time points between the last trigger marker and data sample times
        t = ctx.input_sw.unfold_buffer_t()[-ctx.filter_bank.n_new :]
        idx_end = np.abs(markers_t[trigger_marker_indices[-1], None] - t).argmin(axis=1)

        return markers[trigger_marker_indices[-1]], idx_end

    else:
        # Marker stream got new markers, but none to mark a new epoch
        return None


def get_epoch_data(ctx: ClassifierContext) -> tuple[list[np.ndarray], list[int]]:
    if ctx.current_epos_start != []:
        logger.warning("Tried to extract epochs, but no start marker info present")
        return [], []

    curr_ts = ctx.filter_bank.unfold_buffer_t()

    # buffer index of the first epoch we collected
    mkr_val, idx = ctx.current_epos_start[0]

    # we do not have enough data for the first epoch
    if curr_ts[-1] - curr_ts[idx] < ctx.epo_tmax_s:
        return [], []

    # count how much data we need for the earliest epoch we consider
    tfirst = curr_ts[idx] + ctx.classifier_cfg["tmin"]
    idx_first = np.abs(curr_ts - tfirst).argmin()

    # we only need data from this index onwards -> reflect this by adjusting
    # the n_new of the filter_bank
    ctx.filter_bank.n_new = len(curr_ts) - idx_first

    curr_data = ctx.filter_bank.get_data()

    i = 0
    epochs = []
    markers = []
    while i < len(ctx.current_epos_start):
        # process in FIFO order as first in == earliest
        mkr_val, idx = ctx.current_epos_start[0]

        logger.debug(f"Processing epoch with marker {mkr_val=}, {idx=}")
        # we have enough data for this epoch
        if curr_ts[-1] - curr_ts[idx] > ctx.epo_tmax_s:

            # find index so that data matches the desired epo length most closely
            idx_end = np.abs(curr_ts[idx:] - (curr_ts[idx] + ctx.epo_tmax_s)).argmin()

            logger.debug(f"Enough data for epoch with {idx=}, {idx_end=}")

            epo_x = curr_data[idx:idx_end, :]

            epochs.append(epo_x)
            markers.append(mkr_val)

            # remove this epoch from the list
            ctx.current_epos_start.pop(0)

        i += 1

    return epochs, markers
