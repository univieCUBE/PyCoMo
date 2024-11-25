import warnings
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("pycomo")
log_level = logging.DEBUG
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Logger initialized.')
log_file_name = None


def configure_logger(level=None, log_file=None):
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
    global log_level, log_file_name
    return log_level, log_file_name
