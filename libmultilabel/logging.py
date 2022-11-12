import logging
import transformers.utils.logging as transformer_logging

LOG_FORMAT = '%(asctime)s %(levelname)s:%(message)s'


class ListHandler(logging.Handler):
    """Collect logged message to a list of string
       that can be obtained later.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.logs = []
    
    def emit(self, record):
        string = self.format(record)
        self.logs.append(string)

    def get_logs(self):
        logs = self.logs
        self.logs = []
        return logs

stream_handler = None

def add_stream_handler(level=logging.INFO):
    global stream_handler

    if stream_handler:
        return stream_handler
    else:
        logging.getLogger().setLevel(logging.NOTSET) # use handlers to control levels

        transformer_logging.disable_default_handler()
        transformer_logging.enable_propagation()
        
        lightning_logger = logging.getLogger('pytorch_lightning')
        lightning_logger.handlers.clear()
        lightning_logger.propagate = True

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logging.getLogger().addHandler(stream_handler)
    return stream_handler

collect_handler = None

def add_collect_handler(level=logging.NOTSET):
    global collect_handler

    if collect_handler:
        return collect_handler
    else:
        collect_handler = ListHandler(level=level)
        collect_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        collect_handler.addFilter(lambda record: record.__dict__.get('collect', False))

    logging.getLogger().addHandler(collect_handler)
    return collect_handler



