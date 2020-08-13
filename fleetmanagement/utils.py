""" Helper functions for module """

import os
import logging
import seaborn as sns

def set_logger(name: str = "carnext") -> logging.Logger:
    """
    Returns a formatted logger to use in scripts.

    Args
      name: The name of the logger

    Returns
      logger: A logging.Logger object

    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter(name + " - %(asctime)s - %(message)s", "%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

def set_style():
    sns.set(
        rc={
            "axes.facecolor": "white",
            "axes.labelweight": "bold",
            "axes.labelpad": 40.0,
            "axes.labelsize": 20.0,
            "axes.titlesize": 20.0,
            "figure.figsize": [14.0, 7.0],
            "font.sans-serif": ["Futura", "sans-serif"],
            "font.family": ["Futura", "sans-serif"],
            "grid.alpha": 1.0,
            "grid.color": "black",
            "grid.linestyle": "-",
            "grid.linewidth": 0.1,
            "legend.shadow": False,
            "lines.linewidth": 2.3,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
    sns.set_palette(
        ["#16315a", "#ff4800", "#c4d82e", "#000000", "#8C8C8C", "#27AAE1"]
    )