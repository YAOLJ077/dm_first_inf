"""One continuously fed loky process pool for concurrent fold coordinators.

Use the executor bundled with joblib directly: newly submitted work can start
while earlier tasks are still running. Each fold retains sequential ask/tell.
"""
import os
import threading
from concurrent.futures import Future

from joblib.externals.loky import get_reusable_executor


class JoblibPool:
    def __init__(self, max_workers, initializer=None, initargs=()):
        self.workers = max_workers
        self.lock = threading.RLock()
        self.pending = {}
        self.closed = False
        self.failure = None
        # Match joblib's inner_max_num_threads=1, before child imports numpy.
        env = {name: '1' for name in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
            'NUMEXPR_NUM_THREADS')}
        env['ENABLE_IPC'] = os.environ.get('ENABLE_IPC', '1')
        self.executor = None
        self.executor_options = dict(
            max_workers=max_workers, initializer=initializer, initargs=initargs, env=env)

    def submit(self, function, *args):
        with self.lock:
            if self.failure is not None:
                raise RuntimeError('Joblib dispatcher failed') from self.failure
            if self.closed:
                raise RuntimeError('Joblib pool is closed')
            # Install failure state before waking a waiting coordinator.
            future = Future()
            try:
                if self.executor is None:
                    # Grid uses joblib's reusable executor. Acquire it only on
                    # the first TPE task, after Grid has drained. Replacing its
                    # idle workers avoids leaving a second 64-process pool alive.
                    self.executor = get_reusable_executor(
                        **self.executor_options, reuse=False)
                task = self.executor.submit(function, *args)
            except BaseException as exc:
                self._fail(exc)
                raise
            self.pending[future] = task
            task.add_done_callback(lambda done: self._completed(future, done))
            return future

    def _fail(self, exc):
        self.failure = exc
        pending = list(self.pending.items())
        self.pending.clear()
        for future, task in pending:
            if not future.done():
                future.set_exception(exc)
            task.cancel()

    def _completed(self, future, task):
        with self.lock:
            if task.cancelled():
                self.pending.pop(future, None)
                future.cancel()
                return
            exc = task.exception()
            if exc is not None and self.failure is None:
                self._fail(exc)
                return
            self.pending.pop(future, None)
            if not future.done():
                future.set_result(task.result())

    def shutdown(self, wait=True, cancel_futures=False):
        with self.lock:
            self.closed = True
            if cancel_futures:
                for future, task in list(self.pending.items()):
                    # Running computations finish normally; only cancel work
                    # that the executor has not yet marked as running.
                    if task.cancel():
                        future.cancel()
        # Never hold our lock while joining: completion callbacks need it.
        if self.executor is not None:
            self.executor.shutdown(wait=wait)
