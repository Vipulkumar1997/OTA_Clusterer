import logging

logging.basicConfig(level=logging.INFO)


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger
