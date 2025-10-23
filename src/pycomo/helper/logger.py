import warnings
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import re
import os
import uuid

logging.captureWarnings(True)
# module globals to track the active configured logger
_logger = None
_logger_name = None
log_level = logging.DEBUG
log_file_name = None


# logger = logging.getLogger("pycomo")
# logging.captureWarnings(True)
# log_level = logging.DEBUG
# logger.setLevel(log_level)
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.info('Logger initialized.')
# log_file_name = None


def _make_unique_logger_name(base="pycomo", instance_name=None):
    if instance_name:
        return f"{base}.{instance_name}"
    return f"{base}.{os.getpid()}.{uuid.uuid4().hex[:8]}"


def _ensure_stream_handler(logger_obj, level):
    """Add or update a single StreamHandler on the logger (no duplicates)."""
    for h in logger_obj.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            h.setLevel(level)
            return
    sh = logging.StreamHandler()
    sh.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - PyCoMo - %(levelname)s - %(message)s')
    sh.setFormatter(fmt)
    logger_obj.addHandler(sh)


def _ensure_file_handler(logger_obj, filepath, level):
    """Add a RotatingFileHandler only if not already present for this file."""
    if filepath is None:
        return
    abspath = os.path.abspath(filepath)
    for h in logger_obj.handlers:
        if isinstance(h, RotatingFileHandler):
            # RotatingFileHandler stores baseFilename attribute
            if getattr(h, "baseFilename", None) == abspath:
                h.setLevel(level)
                return
    fh = RotatingFileHandler(abspath, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - PyCoMo - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger_obj.addHandler(fh)


def configure_logger(level=None, log_file=None, instance_name=None, with_name=None):
    """
    Configure the logger with log-level and/or log file location.

    :param level: One of: "debug", "info", "warning", "error"
    :param log_file: Location for the log file
    """
    global _logger, _logger_name, log_level, log_file_name

    log_level_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }

    if level is not None:
        if isinstance(level, str):
            level = level.lower()
            if level in log_level_dict.keys():
                log_level = log_level_dict[level]
            else:
                logger.error(f"Error: Unknown log level string {level}. Use one of {log_level_dict.keys()}. Keeping logger at level {log_level}")
        else:
            log_level = level
        # logger.setLevel(log_level)
        # handler.setLevel(log_level)
        # logger.info(f"Log level set to {level}")

    if log_file is not None:
        log_file_name = log_file
        # file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        # file_handler.setLevel(log_level)
        # file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # file_handler.setFormatter(file_formatter)
        # logger.addHandler(file_handler)
        # logger.info(f"Log file {log_file} added")
    
    if with_name is not None:
        _logger_name = with_name
    elif _logger_name is None:
        _logger_name = _make_unique_logger_name(instance_name=instance_name)
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(log_level)

    # Avoid adding duplicate handlers on repeated configuration calls
    _ensure_stream_handler(_logger, log_level)
    _ensure_file_handler(_logger, log_file_name, log_level)

    _logger.debug(f"Logger '{_logger_name}' initialized (level={log_level}, file={log_file_name})")
    return


def get_logger(name=None):
    """Return the configured logger or a logger by name. If no logger configured, configure default one."""
    global _logger, _logger_name
    if name is not None:
        return logging.getLogger(name)
    if _logger is None:
        configure_logger()
        _logger.debug(f"Created new logger.")
    return _logger


def get_logger_conf():
    """
    Get the current configuration of the logger (logger name, level and file location).

    :return: tuple of logger-name, log-level and log-filename
    """
    global _logger_name, log_level, log_file_name
    if _logger_name is None:
        # ensure at least default configured
        configure_logger()
    return _logger_name, log_level, log_file_name


def get_logger_name():
    """
    Get the current configuration of the logger (logger name, level and file location).

    :return: tuple of logger-name, log-level and log-filename
    """
    global _logger_name, log_level, log_file_name
    if _logger_name is None:
        # ensure at least default configured
        configure_logger()
        _logger.debug(f"Logger name was None. Now {_logger_name}")
    return _logger_name


logger = get_logger(name=_logger_name)
logger.info("Logger initialized.")


class RegexFilter(logging.Filter):
    """
    Filter incoming warning messages with regex patterns. 
    Intended for common warnings of imported packages, that are expected with correct behaviour of PyCoMo.
    """
    def __init__(self, pattern):
        super().__init__()
        self.pattern = re.compile(pattern)

    def filter(self, record):
        # Suppress log messages matching the regex pattern
        return not self.pattern.search(record.getMessage())


# Create a context manager to temporarily add a filter to a logger
@contextmanager
def temporary_logger_filter(logger_name, filter_instance):
    tmp_logger = logging.getLogger(logger_name)
    tmp_logger.addFilter(filter_instance)
    try:
        yield  # Code inside the context block will execute here
    finally:
        tmp_logger.removeFilter(filter_instance)  # Remove the filter afterward


# Suppress warnings about the medium compartment not identifiable by name and
# Adding exchange reaction on sbml import (both are working as intended).
regex_filter_medium = RegexFilter(r"Could not identify an external compartment by name ")
regex_filter_sbml = RegexFilter(r"Adding exchange reaction ")
logging.getLogger("cobra.medium.boundary_types").addFilter(regex_filter_medium)
logging.getLogger("cobra.io.sbml").addFilter(regex_filter_sbml)
