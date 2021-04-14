import datetime
import json
import logging
import os
import time
import traceback
from functools import wraps

DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.basicConfig(
    level=INFO,
    format='%(levelname)s:%(name)s: %(message)s',
)


def get_current_time_utc_iso():
    # add trailing Z to match iso format
    return datetime.datetime.utcnow().isoformat() + 'Z'


def _use_new_value(old, new):
    return new


def _sum_with(old, new):
    if old is None:
        old = 0
    return old + new


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    app_insight_tracer = None

    def __init__(self):
        self.loggers = []
        self.paths = []
        self.app_insight_spans = []
        self.collectors_list = []
        self.level = INFO
        self.start_time = get_time()
        self.last_time = self.start_time
        self.operation_id = None

        # base level
        base_name = os.environ.get('LOG_ROOT', 'med_ai')
        base_name = os.environ.get('LOG_ROOT', '.')
        self.paths.append(base_name)
        logger = logging.getLogger(base_name)
        logger.setLevel(self.level)
        self.loggers.append(logger)

    def get_path(self):
        return '.'.join(self.paths)

    def enter(self, name, record_span=True):
        """decorator of function: enter another logger domain.

        Example:
            @enter('bar')
            def bar_func():
                log.info('inside bar')

            @enter('foo')
            def foo_func():
                bar_func()

            should log like

            bar.foo: inside foo

        Args:
            name (str): The name of this sub logger.
            record_span (bool): To record span info to local or app insight. Defaults to False.
        """
        def enter_func_decorate(func):
            @wraps(func)
            def actual_func(*args, **kwargs):
                with LogLayer(name, record_span):
                    return func(*args, **kwargs)
            return actual_func
        return enter_func_decorate

    def get_logger(self):
        return self.loggers[-1]

    def record_collectors(self, key, msg, is_list, cb=_use_new_value):
        if not isinstance(msg, (str, int, float)):
            msg = json.dumps(msg)
        for collectors in self.collectors_list:
            if is_list:
                collectors.setdefault(key, [])
                collectors[key].append(msg)
            else:
                collectors[key] = cb(collectors.get(key), msg)

    def log(self, log_key, msg):
        assert log_key in set(['info', 'warning', 'debug', 'error'])
        getattr(self.get_logger(), log_key)(msg)
        self.record_collectors(
            'logs',
            {
                'level': log_key,
                'path': self.get_path(),
                'message': f'{msg}',
                'time': get_current_time_utc_iso(),
            },
            True,
        )

    def log_extra(self, log_key: str, value, is_list=False, cb=None):
        '''Log extra fields (to app insight)

        Args:
            log_key (str): The key in customEvents.customDimensions.
            value (any): Logged value. Will parsed by `str`.
            is_list (bool, optional): Is it logged as a list or just an object. Defaults to False.
            cb (func, optional): cb(old, new) for updating logged value (if not is_list).
                Defaults to lambda old, new: new (Use new value)
        '''
        assert log_key not in set(['info', 'warning', 'debug', 'error'])
        if not cb:
            cb = _use_new_value
        self.record_collectors(log_key, value, is_list, cb)


class LogLayer(object):
    def __init__(self, name, record_span):
        self.name = name
        self.span = None
        self.record_span = record_span
        self.time = None

    def __enter__(self):
        Logger().paths.append(self.name)
        logger = logging.getLogger(Logger().get_path())
        logger.setLevel(Logger().level)
        Logger().loggers.append(logger)

        if self.record_span:
            self.time = get_time()

    def __exit__(self, type, value, traceback):
        paths = Logger().get_path()
        Logger().paths.pop()
        Logger().loggers.pop()

        if self.record_span:
            time_diff = int((get_time() - self.time) * 1000)
            extra(f"dur_{paths.replace('.', '__')}", time_diff, _sum_with)


class LogCollector():
    def __init__(self):
        self.logs = {}

    def __enter__(self):
        Logger().collectors_list.append(self.logs)

    def __exit__(self, type, value, traceback):
        Logger().collectors_list.pop()


def enter(*args, **kwargs):
    return Logger().enter(*args, **kwargs)


def info(msg: str):
    Logger().log('info', msg)


def debug(msg: str):
    Logger().log('debug', msg)


def warning(msg: str):
    Logger().log('warning', msg)


def error(msg: str):
    Logger().log('error', msg)


def extra(key, msg, cb=None):
    Logger().log_extra(key, msg, False, cb)


def _get_format_ext():
    return traceback.format_exc()


def get_time():
    '''To make time mock-able when test'''
    return time.time()
