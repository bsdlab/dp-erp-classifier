# The main event loop
import time
from threading import Event, Thread

import ujson
from dareplane_utils.general.event_loop import EventLoop

from erp_classifier.context import ClassifierContext, get_context
from erp_classifier.feature_extraction import get_epoch_data, new_epoch_started
from erp_classifier.logging import logger


def classification_callback(ctx: ClassifierContext):

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

    ctx.filter_bank.n_new -= n_added  # adjust to the start of the latest epoch added

    for epo in ctx.epochs_for_clf_stack:
        # get feature vector and store for online adaptation
        vec = ctx.vectorizer.transform(epo.data)
        ctx.feature_vecs.add_samples([vec], [time.perf_counter()])

        # predict
        pred = ctx.clf_pipeline.predict(vec)

        # send to LSL
        ctx.outlet.push_sample(ujson.dumps({"pred": pred.tolist()}))


def run_online_classifier(stop_event: Event):
    """
    Run the online classifier until the stop event is set.

    Parameters
    ----------
    stop_event : Event
        An event to signal when to stop the classifier.
    """
    ctx = get_context()
    ctx.connect_input()

    ev = EventLoop(dt_s=ctx.dt_s, stop_event=stop_event, ctx=ctx)

    ev.add_callback(classification_callback)

    ev.run()


def run_online_classifier_in_thread() -> tuple[Thread, Event]:
    """Run the online classifier in a separate thread. Returns the thread and the stop event"""
    stop_event = Event()
    thread = Thread(target=run_online_classifier, args=(stop_event,))
    thread.start()
    return thread, stop_event
