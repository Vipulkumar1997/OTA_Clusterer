import logging

logging.basicConfig(level=logging.INFO)


def get_logger():
    """provides an logger object instance to other modules
    :return: logger

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger
