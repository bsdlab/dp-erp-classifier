import time

import numpy as np
import pytest

from erp_classifier.context import get_context
from erp_classifier.feature_extraction import get_new_epochs
from erp_classifier.logging import logger
from tests.shared import ctx, spawn_lsl_marker_stream, spawn_lsl_stream

logger.setLevel("INFO")


def test_epoching(ctx, spawn_lsl_stream, spawn_lsl_marker_stream):

    ctx.decode_trigger_markers = [101, 102, 103, 104]
    ctx.connect_input(stream_name="test", marker_stream_name="markers")

    start = time.perf_counter()

    while time.perf_counter() - start < 2:

        # get new data into the StreamWatchers and pass through the filter_bank
        ctx.update_stream_watchers()

        ctx.epochs_accumulating_stack.extend(get_new_epochs(ctx))

        n_added = 0
        curr_data = None
        curr_ts = None

        # check if we already have completed epochs
        while ctx.epochs_accumulating_stack != []:
            if curr_data is None and curr_ts is None:
                # logger.debug(f"Looking at {ctx.filter_bank.n_new=}")

                # time samples in the current window of interest
                curr_ts = ctx.filter_bank.ring_buffer.unfold_buffer_t()[
                    -ctx.filter_bank.n_new :
                ]

                curr_data = ctx.filter_bank.get_data()[-ctx.filter_bank.n_new :]

            epo = ctx.epochs_accumulating_stack[0]

            # if enough data -> fill, pop epoch, and add to the classifier stack
            if (epo.ts_event + epo.tmax) < curr_ts[-1]:

                # fill data to the epoch from the closests to the tmin
                idx = np.abs(curr_ts - (epo.ts_event + epo.tmin)).argmin()
                epo.add_data(
                    curr_data[
                        idx : idx + epo.nsamples, :, 0
                    ],  # only one freq band so take 0 for last dim
                    curr_ts[idx : idx + epo.nsamples],
                )

                ctx.epochs_for_clf_stack.append(ctx.epochs_accumulating_stack.pop(0))
                n_added = max(n_added, idx + 1)

            # end time for first epoch not yet reached, nothing to do for now
            else:
                break

        ctx.filter_bank.n_new -= (
            n_added  # adjust to the start of the latest epoch added
        )

        # Not subject of this test -> here would be the decoding part
        for epo in ctx.epochs_for_clf_stack:
            pass

        time.sleep(ctx.dt_s)

    # we should have 3 full epochs at this point
    assert len(ctx.epochs_for_clf_stack) == 3
    assert (
        np.abs(np.diff([e.ts_event for e in ctx.epochs_for_clf_stack]).mean() - 0.25)
        < 0.05
    )
    # marker stream for testing is created to have consecutive markers always with diff = +/-1
    assert (np.abs(np.diff([e.event for e in ctx.epochs_for_clf_stack])) == 1).all()
