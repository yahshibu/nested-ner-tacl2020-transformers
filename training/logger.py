import logging
import sys
from _io import TextIOWrapper
from logging import Logger


def get_logger(name: str, level: int = logging.INFO, stream: TextIOWrapper = sys.stdout, file: str = None,
               formatter: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(formatter)

    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
