import warnings
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import re

logger = logging.getLogger("pycomo")
logging.captureWarnings(True)
log_level = logging.DEBUG
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Logger initialized.')
log_file_name = None


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


def configure_logger(level=None, log_file=None):
    """
    Configure the logger with log-level and/or log file location.

    :param level: One of: "debug", "info", "warning", "error"
    :param log_file: Location for the log file
    """
    global log_level, log_file_name

    log_level_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }

    if level is not None:
        if isinstance(level, str):
            if level.lower() in log_level_dict.keys():
                log_level = log_level_dict[level.lower()]
            else:
                logger.error(f"Error: Unknown log level string {level}. Use one of {log_level_dict.keys()}")
        else:
            log_level = level
        logger.setLevel(log_level)
        handler.setLevel(log_level)
        logger.info(f"Log level set to {level}")

    if log_file is not None:
        log_file_name = log_file
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file {log_file} added")
    return


def get_logger_conf():
    """
    Get the current configuration of the logger (level and file location).

    :return: tuple of log-level and log-filename
    """
    global log_level, log_file_name
    return log_level, log_file_name
