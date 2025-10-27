"""
Test script for varying latent dimensions and simulations sets in a neural network.
"""
import os
from typing import Any

import numpy as np
from numpy import ndarray

from src.batch_train import update_sims
from src.utils.utils import open_config
from src.tests.netloader_tests import TestConfig, run_tests


def change_latent_dim(
        config: dict[str, dict[str, Any] | list[dict[str, Any]]],
        *,
        latent_dim: int) -> None:
    """
    Change the latent dimension in the network JSON file given by the layer before the last
    Checkpoint layer.

    Parameters
    ----------
    config : dict[str, dict[str, Any] | list[dict[str, Any]]]
        The network configuration file
    latent_dim : int
        The new latent dimension to set
    """
    i: int = 0
    layer: dict[str, Any]

    for i, layer in enumerate(config['layers'][::-1]):
        if layer['type'] == 'Checkpoint':
            break

    config['layers'][-i - 2]['features'] = latent_dim


def generate_test_configs(
        repeats: int,
        net_name: str,
        data_dir: str,
        latent_dims: list[int],
        sim_sets: list[list[str]],
        cumulative: bool = False,
        description: str = '',
        hyperparams: dict[str, Any] | None = None) -> list[TestConfig]:
    """
    Generate a list of test configurations for different latent dimensions and simulation sets.

    Parameters
    ----------
    repeats : int
        Number of times to repeat each test configuration
    net_name : str
        Network configuration file name
    data_dir : str
        Path to the data directory
    latent_dims : list[int]
        List of latent dimensions to test
    sim_sets : list[list[str]]
        List of simulation sets to test
    cumulative : bool, default = False
        If simulations should be cumulatively added
    description : str, default = ''
        Description of the tests
    hyperparams : dict[str, Any] | None, default = None
        Optional hyperparameters for the tests

    Returns
    -------
    list[TestConfig]
        List of test configurations
    """
    tests = []
    current_known: ndarray = np.array([])

    for i, sims in enumerate(sim_sets):
        if cumulative:
            current_known = update_sims(sims, current_known)
        else:
            current_known = np.array(sims)

        for j, latent_dim in enumerate(latent_dims):
            for k in range(repeats):
                tests.append(
                    TestConfig(
                        f'{i}.{j}.{k}',
                        f'{description}\n' if description else ''
                        f'Latent Dim: {latent_dim}\nSims: {current_known}',
                        {'data_dir': data_dir, 'sims': current_known.tolist(), 'unknown_sims': []},
                        net_name,
                        network_mod_params={'latent_dim': latent_dim},
                        hyperparams=hyperparams,
                        network_mod_fn=change_latent_dim,
                    )
                )
    return tests


def main(config_path: str = '../config.yaml') -> None:
    """
    Main function to run the latent dimension tests.

    Parameters
    ----------
    config_path : str, default = '../config.yaml'
        Path to the configuration file
    """
    repeats: int = 3
    epochs: int = 150
    save: int | str = 'latent_1'
    load: int | str = save
    latent_dims: list[int] = np.concat((
        np.arange(1, 10),
        np.arange(10, 22, 2),
    )).tolist() + [50, 100, 1000]
    sim_sets: list[list[str]] = [
        ['bahamas_cdm', 'bahamas_0.1', 'bahamas_0.3', 'bahamas_1'],
        ['bahamas_cdm_low', 'bahamas_cdm_hi'],
        ['darkskies_cdm', 'darkskies_0.1', 'darkskies_0.2'],
        ['flamingo', 'flamingo_low', 'flamingo_hi'],
    ]
    config: dict[str, Any]
    hyperparams: dict[str, Any] = {'epochs': epochs}

    _, config = open_config('main', config_path)
    tests = generate_test_configs(
        repeats,
        'network_v10',
        config['data']['data-dir'],
        latent_dims,
        sim_sets,
        hyperparams=hyperparams,
        cumulative=True,
    )
    run_tests(
        save,
        load,
        os.path.join(config['data']['data-dir'], f'{save}.pkl'),
        tests,
        config=config,
    )


if __name__ == '__main__':
    main()
