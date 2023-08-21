from typing import Optional
import logging


def setup_logging(logfile: Optional[str] = None, level=logging.INFO) -> Optional[logging.FileHandler]:
    """
    Setup logging by defining the fromat and handlers

    :param logfile: Path to logfile, if None only logs to console, defaults to None
    :type logfile: Optional[str], optional
    :param level: Minimum level to log, defaults to logging.INFO
    :type level: [TODO:type], optional
    :return: Filehandler if logging to logfile, otherwise None. Can be used to close handler
    :rtype: Optional[logging.FileHandler]
    """
    logging.basicConfig(level=level)
    root_logger = logging.getLogger()
    root_logger.handlers = []
    format = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s.%(funcName)s:%(message)s")
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(format)
    root_logger.addHandler(stdout_handler)
    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(format)
        root_logger.addHandler(file_handler)
        return file_handler
    return None
