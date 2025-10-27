"""
Misc functions used elsewhere
"""
import os
from typing import Any
from argparse import ArgumentParser

import yaml
import numpy as np
from scipy.stats import gaussian_kde


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _interactive_check() -> bool:
    """
    Checks if the launch environment is interactive or not

    Returns
    -------
    bool
        If environment is interactive
    """
    if os.getenv('PYCHARM_HOSTED'):
        return True

    try:
        if get_ipython().__class__.__name__:
            return True
    except NameError:
        return False

    return False


def open_config(
        key: str,
        config_path: str,
        parser: ArgumentParser | None = None) -> tuple[str, dict[str, Any]]:
    """
    Opens the configuration file from either the provided path or through command line argument

    Parameters
    ----------
    key : str
        Key of the configuration file
    config_path : str
        Default path to the configuration file
    parser : ArgumentParser | None, default = None
        Parser if arguments other than config path are required

    Returns
    -------
    tuple[str, dict[str, Any]]
        Configuration path and configuration file dictionary
    """
    if not _interactive_check():
        if not parser:
            parser = ArgumentParser()

        parser.add_argument(
            '--config_path',
            default=config_path,
            help='Path to the configuration file',
            required=False,
        )
        args = parser.parse_args()
        config_path = args.config_path

    config_path += '' if '.yaml' in config_path else '.yaml'

    with open(os.path.join(ROOT, config_path), 'rb') as file:
        config = yaml.safe_load(file)[key]

    return config_path, config


def overlap(data_1: np.ndarray, data_2: np.ndarray, bins: int = 100) -> float:
    """
    Calculates the overlap between two datasets by using a Gaussian kernel to approximate the
    distribution, then integrates the overlap using the trapezoidal rule

    Parameters
    ----------
    data_1 : ndarray
        First dataset of shape (N), where N are the number of points
    data_2 : ndarray
        Second dataset of shape (M), where M are the number of points
    bins : int, default = 100
        Number of bins to sample from the Gaussian distribution approximation

    Returns
    -------
    float
        Overlap fraction
    """
    grid = np.linspace(min(data_1.min(), data_2.min()), max(data_1.max(), data_2.max()), bins)
    kde_1 = gaussian_kde(data_1)
    kde_2 = gaussian_kde(data_2)

    pdf_1 = kde_1(grid)
    pdf_2 = kde_2(grid)
    return np.trapezoid(np.minimum(pdf_1, pdf_2), grid)
