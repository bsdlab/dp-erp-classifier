# The main event loop
from threading import Event, Thread

import ujson
from dareplane_utils.general.event_loop import EventLoop

from erp_classifier.context import ClassifierContext, get_context
from erp_classifier.feature_extraction import get_epoch_data, new_epoch_started
from erp_classifier.logging import logger


def update_stream_watchers(ctx: ClassifierContext):
    """Update the StreamWatchers and check for new data"""
    ctx.input_sw.update()
    ctx.input_mrk_sw.update()

    # If new data -> filter it
    if ctx.input_sw.n_new > 0:
        x = ctx.input_sw.unfold_buffer()[-ctx.input_sw.n_new :]
        ts = ctx.input_sw.unfold_buffer_t()[-ctx.input_sw.n_new :]
        ctx.input_sw.n_new = 0
        ctx.filter_bank.filter(x, ts)


def classification_callback(ctx: ClassifierContext):

    update_stream_watchers(ctx)

    # check if marker has arrived for new epoch, if so, we remember add
    # the marker values and the index with the closest time stamp in the filter buffer
    epo_start = new_epoch_started(
        ctx
    )  # for now this only warns if there are multiple epochs started in the current chunk, potentially process all (but ensure a minimal time distance potentially)
    if epo_start is not None:
        ctx.current_epos_start.append(epo_start)

    if ctx.current_epos_start == []:
        pass  # early return as nothing to do
    else:
        epochs_x, markers = get_epoch_data(ctx)
        for epo, mrk in zip(epochs_x, markers):
            features = ctx.vectorizer.transform(epo)
            pred = ctx.clf_pipeline.predict(features)
            # send the prediction to the output stream
            res = {"pred": pred, "features": features, "marker": mrk}
            logger.debug(f"Sending prediction: {res=}")
            ctx.outlet.push_sample([ujson.dumps(res)])


def run_online_classifier(stop_event: Event):
    """
    Run the online classifier until the stop event is set.

    Parameters
    ----------
    stop_event : Event
        An event to signal when to stop the classifier.
    """
    ctx = get_context()
    ev = EventLoop(dt_s=ctx.dt_s, stop_event=stop_event, ctx=ctx)

    ev.add_callback(classification_callback)

    ev.run()


def run_online_classifier_in_thread() -> tuple[Thread, Event]:
    """Run the online classifier in a separate thread. Returns the thread and the stop event"""
    stop_event = Event()
    thread = Thread(target=run_online_classifier, args=(stop_event,))
    thread.start()
    return thread, stop_event
