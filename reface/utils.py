import threading
import cv2
import numpy as np
from queue import Queue
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import Future
from types import SimpleNamespace
from dacite import types as typing_utils
from reface.config import Config


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
                future.add_done_callback(self._done_futures.put)
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


def map_fast(
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


def draw_bbox(image, bbox):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 3)


def load_config_from_dict(dictionary) -> Config:
    # actually it returns SimpleNamespace, but we want to fool type hinter
    # so it will prompt us the config fields in auto-completion

    def load_from_template(template, cfg_value):
        if isinstance(cfg_value, dict):
            cfg_obj = SimpleNamespace()
            for key, value in cfg_value.items():
                nest_template = getattr(template, key, None)
                if nest_template is not None:
                    # it is a nested class which means subsection of config
                    result = load_from_template(nest_template, value)
                    setattr(cfg_obj, key, result)
                else:
                    # it should be either raw value or a Union
                    anno = template.__annotations__[key]
                    if typing_utils.is_union(anno):
                        nest_templates = typing_utils.extract_generic(anno)
                        result = load_from_matching_template(nest_templates, value)
                        setattr(cfg_obj, key, result)
                    else:
                        setattr(cfg_obj, key, value)
            return cfg_obj
        elif isinstance(cfg_value, list):
            if typing_utils.is_generic_collection(template):
                item_template = typing_utils.extract_generic(template)[0]
                if typing_utils.is_union(item_template):
                    item_templates = typing_utils.extract_generic(item_template)
                else:
                    item_templates = [item_template]
                return [
                    load_from_matching_template(item_templates, item)
                    for item in cfg_value
                ]
            else:
                return cfg_value

        else:
            return cfg_value

    def load_from_matching_template(templates, cfg_value):
        for template in templates:
            try:
                return load_from_template(template, cfg_value)
            except KeyError:
                pass
        raise KeyError("none of templates matched", templates)

    # noinspection PyTypeChecker
    return load_from_template(Config, dictionary)


def image_from_torch(image):
    image = image.cpu().numpy()
    return np.transpose(image.astype("uint8"), (1, 2, 0))
