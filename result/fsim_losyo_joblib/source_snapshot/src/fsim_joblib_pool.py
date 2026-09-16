"""Single-owner joblib.Parallel pool serving concurrent calibration coordinators.

Only the dispatcher calls Parallel. Short, bounded batches avoid sharing a
non-reentrant Parallel instance across threads or creating one pool per fold.
"""
import queue
import threading
import time
from concurrent.futures import Future

from joblib import Parallel, delayed, parallel_config


def execute_tagged(index, function, args):
    return index, function(*args)


class JoblibPool:
    def __init__(self, max_workers, initializer=None, initargs=()):
        self.workers = max_workers
        self.initializer, self.initargs = initializer, initargs
        self.queue = queue.Queue()
        self.lock = threading.RLock()
        self.pending = set()
        self.closed = False
        self.failure = None
        self.owner = threading.Thread(target=self._dispatch, name='joblib-dispatch', daemon=True)
        self.owner.start()

    def submit(self, function, *args):
        with self.lock:
            if self.failure is not None:
                raise RuntimeError('Joblib dispatcher failed') from self.failure
            if self.closed:
                raise RuntimeError('Joblib pool is closed')
            future = Future()
            self.pending.add(future)
            self.queue.put((future, function, args))
            return future

    def _dispatch(self):
        try:
            # n_jobs=1 runs in this thread and joblib does not invoke initializer.
            if self.workers == 1 and self.initializer is not None:
                self.initializer(*self.initargs)
            with parallel_config(backend='loky', inner_max_num_threads=1):
                with Parallel(n_jobs=self.workers, return_as='generator_unordered',
                              batch_size=1, pre_dispatch=self.workers,
                              initializer=self.initializer, initargs=self.initargs) as parallel:
                    while True:
                        first = self.queue.get()
                        if first is None:
                            return
                        batch = [first]
                        deadline = time.monotonic() + 0.01
                        finish = False
                        while len(batch) < self.workers * 2:
                            try:
                                item = self.queue.get(timeout=max(0, deadline - time.monotonic()))
                            except queue.Empty:
                                break
                            if item is None:
                                finish = True
                                break
                            batch.append(item)
                        active = []
                        for future, function, args in batch:
                            if future.set_running_or_notify_cancel():
                                active.append((future, function, args))
                            else:
                                with self.lock:
                                    self.pending.discard(future)
                        for index, result in parallel(delayed(execute_tagged)(i, fn, args)
                                                       for i, (_, fn, args) in enumerate(active)):
                            future = active[index][0]
                            future.set_result(result)
                            with self.lock:
                                self.pending.discard(future)
                        if finish:
                            return
        except BaseException as exc:
            with self.lock:
                self.failure = exc
                for future in self.pending:
                    if not future.done():
                        future.set_exception(exc)
                self.pending.clear()

    def shutdown(self, wait=True, cancel_futures=False):
        with self.lock:
            if not self.closed:
                self.closed = True
                if cancel_futures:
                    for future in self.pending:
                        future.cancel()
                self.queue.put(None)
        if wait:
            self.owner.join()
