import logging

LOG_FORMAT = "%(asctime)s %(levelname)s:%(message)s"


class ListHandler(logging.Handler):
    """Collect logged messages to a list of strings that can be obtained later.
    The `logging` module does not provide this function, so we implement one.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.logs = []

    def emit(self, record):
        string = self.format(record)
        self.logs.append(string)

    def get_logs(self):
        """Return and clear all collected logs.

        Returns:
            list[str]: A list of formatted logs.
        """
        logs = self.logs
        self.logs = []
        return logs


stream_handler = None


def add_stream_handler(level=logging.INFO):
    """Create and return a stream handler so that logging messages are
    sent to the terminal. The stream handler is attached to the root logger.
    The logging messages from the `transformer` and `pytorch_lighting`
    modules are propagated to the root logger so they can be managed by
    us (e.g., silence them in silent mode).
    If the handler had been created, this function returns the handler
    created earlier instead.

    Returns:
        logging.StreamHandler: The created stream handler.
    """
    global stream_handler

    if stream_handler:
        return stream_handler
    else:
        logging.getLogger().setLevel(logging.NOTSET)  # use handlers to control levels

        try:
            import transformers.utils.logging as transformer_logging

            transformer_logging.disable_default_handler()
            transformer_logging.enable_propagation()
        except ImportError:
            pass

        lightning_logger = logging.getLogger("pytorch_lightning")
        lightning_logger.handlers.clear()
        lightning_logger.propagate = True

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(stream_handler)

    return stream_handler


collect_handler = None


def add_collect_handler(level=logging.NOTSET):
    """Create and return a ListHandler so that logging records with the attribute
    `collect=True` is collected. The ListHandler is attached to the root logger.
    If the handler had been created, this function returns the handler created
    earlier instead.

    To collect a log, set the key 'collect' with the value `True` in the `extra`
    argument when logging. An example (similarly for logs of other levels):

        logging.info('important message', extra={'collect': True})

    Returns:
        ListHandler: The created ListHandler.
    """
    global collect_handler

    if collect_handler:
        return collect_handler
    else:
        logging.getLogger().setLevel(logging.NOTSET)  # use handlers to control levels

        collect_handler = ListHandler(level=level)
        collect_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        collect_handler.addFilter(lambda record: record.__dict__.get("collect", False))
        logging.getLogger().addHandler(collect_handler)

    return collect_handler
