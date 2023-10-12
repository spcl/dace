from typing import Optional, List
import logging
import os

logger = logging.getLogger(__name__)


def setup_logging(
        logfile: Optional[str] = None,
        full_logfile: Optional[str] = None,
        level=logging.INFO) -> List[logging.FileHandler]:
    """
    Setup logging by defining the fromat and handlers. 

    :param logfile: Path to logfile, if None only logs to console, defaults to None
    :type logfile: Optional[str], optional
    :param full_logfile: Path to logfile with level DEBUG, if None no such logfile is created, defaults to None
    :type logfile: Optional[str], optional
    :param level: Minimum level to log, defaults to logging.INFO. This affects the console output and logfile.
    :type level: [TODO:type], optional
    :return: Filehandlers if logging to logfile. Otherwise an empty list. Can be used to close handler
    :rtype: List[logging.FileHandler]
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    format = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s.%(funcName)s: %(message)s")
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(format)
    stdout_handler.setLevel(level)
    root_logger.addHandler(stdout_handler)
    file_handlers = []
    if full_logfile is not None:
        if os.path.exists(full_logfile):
            os.remove(full_logfile)
        full_file_handler = logging.FileHandler(full_logfile)
        full_file_handler.setFormatter(format)
        full_file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(full_file_handler)
        file_handlers.append(full_file_handler)
    if logfile is not None:
        if os.path.exists(logfile):
            os.remove(logfile)
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(format)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        file_handlers.append(file_handler)
        logger.debug("Create file handler using file at %s and level %s", logfile, level)
    if full_logfile:
        logger.debug("Create file handler using file at %s for full logfile", full_logfile)
    logger.debug("Set level of console logger to %s", level)
    return file_handlers


def close_filehandlers(file_handlers: List[logging.FileHandler]):
    """
    Closes and flushes given filehandlers

    :param file_handlers: The filehandlers to close
    :type file_handlers: List[logging.FileHandler]
    """
    for fh in file_handlers:
        fh.flush()
        fh.close()
