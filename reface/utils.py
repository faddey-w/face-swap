import threading
from queue import Queue
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import Future


class Mapper:
    def __init__(self, function, *iterables):
        self._function = function
        self._iterables = iterables
        self._cancelled = False

    def __iter__(self):
        for result in map(self._function, *self._iterables):
            yield result
            if self._cancelled:
                raise StopIteration

    def __len__(self):
        return _get_len(self._iterables)

    def cancel(self):
        self._cancelled = True


class AsyncMapUnordered(Mapper):
    def __init__(self, function, *iterables, num_workers=None, buffer_size=None):
        super(AsyncMapUnordered, self).__init__(function, *iterables)
        self._executor = ThreadPoolExecutor(num_workers)
        if not buffer_size:
            buffer_size = self._executor._max_workers * 3
        self._sender_thread = None
        self._done_futures = Queue(buffer_size)
        self._cancelled = threading.Event()
        self._finished = False
        self._cancel_marker = object()

    def __iter__(self):
        return self

    def __next__(self):
        if self._sender_thread is None:
            self._sender_thread = threading.Thread(
                target=self._sender_main, daemon=True
            )
            self._sender_thread.start()
        if self._finished:
            raise StopIteration
        future = self._done_futures.get()
        try:
            if future is None:
                self._finished = True
                raise StopIteration
            result = future.result()
        except BaseException:
            self._finished = True
            raise
        else:
            if result is self._cancel_marker:
                self._finished = True
                raise StopIteration
            else:
                return result
        finally:
            if self._finished:
                self.cancel()
                self._executor.shutdown(wait=True)

    def cancel(self):
        self._cancelled.set()

    def _sender_main(self):
        n_in_progress = 0
        all_sent = False
        lock = threading.Lock()

        def update_in_progress(future):
            nonlocal n_in_progress
            del future
            with lock:
                n_in_progress -= 1
                if all_sent and n_in_progress == 0:
                    self._done_futures.put(None)

        try:
            for args in zip(*self._iterables):
                if self._cancelled.isSet():
                    break
                future = self._executor.submit(self._call_function, *args)
                with lock:
                    n_in_progress += 1
                future.add_done_callback(
                    self._done_futures.put
                )
                future.add_done_callback(update_in_progress)
                del future

        except BaseException as exc:
            error_future = Future()
            error_future.set_exception(exc)
            self._done_futures.put(error_future)
        finally:
            with lock:
                all_sent = True

    def _call_function(self, *args):
        if self._cancelled.isSet():
            return self._cancel_marker
        return self._function(*args)


def async_map_unordered(
    function, *iterables, size_threshold=100, num_workers=None, buffer_size=None
):
    need_async = True
    if size_threshold is not None:
        try:
            input_len = _get_len(iterables)
        except TypeError:
            input_len = None
        if input_len is not None:
            if input_len <= size_threshold:
                need_async = False
    if need_async:
        mapper = AsyncMapUnordered(
            function, *iterables, num_workers=num_workers, buffer_size=buffer_size
        )
    else:
        mapper = Mapper(function, *iterables)
    return mapper


def _get_len(iterables):
    return min(map(len, iterables))