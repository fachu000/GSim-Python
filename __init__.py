from .experiment_set import AbstractExperimentSet
from .experiment_set import OUTPUT_DATA_FOLDER
import logging

from .gfigure import GFigure

from .utils import time_to_str, print_obj_attributes, instr

from numpy.random import default_rng

rng = default_rng()


def init_gsim_logger():

    class ColoredFormatter(logging.Formatter):
        # Define color codes
        COLORS = {
            'DEBUG': '\033[94m',  # Blue
            'INFO': '\033[92m',  # Green
            'WARNING': '\033[93m',  # Yellow
            'ERROR': '\033[91m',  # Red
            'CRITICAL': '\033[95m',  # Magenta
        }
        RESET = '\033[0m'  # Reset color

        def format(self, record):
            # Add color to the log level name
            levelname = record.levelname
            if levelname in self.COLORS:
                levelname_color = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
                record.levelname = levelname_color
            return super().format(record)

    logger = logging.getLogger("gsim")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = ColoredFormatter('{levelname}:{module}:{lineno}: {message}',
                                 style='{')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
