"""Logging mogdule."""
import logging
from datetime import datetime

from emonet import ROOT

LOG_DIR = ROOT.joinpath("logs")
LOG_DIR.mkdir(exist_ok=True)


def make_logger(name: str) -> logging.Logger:
    """
    Create a logger object.

    Parameters
    ----------
    name: str
        Return a logger with the specified name; or, if name is None, return a logger which is the
        root logger of the hierarchy.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        f'{LOG_DIR}/model_run_{datetime.today().strftime("%Y-%m-%d-%H-%M")}.log'
    )
    handler.setFormatter(logging.Formatter("%(levelname)s::%(name)s::%(message)s"))
    logger.addHandler(handler)
    return logger
