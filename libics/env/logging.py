import logging
import sys

from logging import (
    DEBUG, INFO, WARNING, ERROR, CRITICAL
)


###############################################################################


class MaxFilter(logging.Filter):

    """
    Logging filter specifying the maximum log level to be handled.
    """

    def __init__(self, maxlevel=100):
        self.maxlevel = maxlevel

    def filter(self, record):
        return record.levelno <= self.maxlevel


###############################################################################


LOG_FMT = logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)


DEBUG_HANDLER = logging.StreamHandler(stream=sys.stdout)
DEBUG_HANDLER.setFormatter(LOG_FMT)
DEBUG_HANDLER.setLevel(DEBUG)
DEBUG_HANDLER.addFilter(MaxFilter(maxlevel=INFO-1))
INFO_HANDLER = logging.StreamHandler(stream=sys.stdout)
INFO_HANDLER.setFormatter(LOG_FMT)
INFO_HANDLER.setLevel(INFO)
INFO_HANDLER.addFilter(MaxFilter(maxlevel=WARNING-1))
WARNING_HANDLER = logging.StreamHandler(stream=sys.stdout)
WARNING_HANDLER.setFormatter(LOG_FMT)
WARNING_HANDLER.setLevel(WARNING)
WARNING_HANDLER.addFilter(MaxFilter(maxlevel=ERROR-1))
ERROR_HANDLER = logging.StreamHandler(stream=sys.stderr)
ERROR_HANDLER.setFormatter(LOG_FMT)
ERROR_HANDLER.setLevel(ERROR)
ERROR_HANDLER.addFilter(MaxFilter(maxlevel=CRITICAL-1))
CRITICAL_HANDLER = logging.StreamHandler(stream=sys.stderr)
CRITICAL_HANDLER.setFormatter(LOG_FMT)
CRITICAL_HANDLER.setLevel(CRITICAL)
HANDLERS = [
    DEBUG_HANDLER, INFO_HANDLER,
    WARNING_HANDLER, ERROR_HANDLER, CRITICAL_HANDLER
]


###############################################################################


def get_logger(name, level=WARNING):
    """
    Gets a named logger instance.

    Parameters
    ----------
    name : `str`
        Name of logger. Should be hierarchically dot-separated.
    level : `int`
        Default log level.
        Use one of `DEBUG, INFO, WARNING, ERROR, CRITICAL`.

    Examples
    --------
    >>> logger = get_logger("libics.core.my_module", level=logging.INFO)
    >>> logger.error("some error occured")
    2020-02-02 20:20,002 [libics.core.my_module] ERROR: some error occured
    >>> logger.debug("some debug message")
    >>> logger.setLevel(logging.DEBUG)
    >>> logger.debug("some debug message")
    2020-02-02 20:20,200 [libics.core.my_module] DEBUG: some debug message
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    [logger.addHandler(h) for h in HANDLERS]
    return logger
