"""
Functions to run tests on neural networks using the PyTorch-Network-Loader package.
"""
import os
import json
import pickle
from dataclasses import dataclass
from typing import Callable, Any, TextIO

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from netloader.networks import BaseNetwork
from netloader.utils.utils import save_name
from netloader.data import loader_init, BaseDataset

from src.main import net_init
from src.utils.data import DarkDataset
from src.utils.utils import open_config, ROOT


@dataclass
class TestConfig:
    # pylint: disable=line-too-long
    """
    Configuration for a test to be run on the neural network.

    Attributes
    ----------
    name : str
        Name of the test
    description : str
        Description of the test
    dataset_args : dict[str, Any]
        Keyword arguments for the dataset initialization
    net_name : str
        Name of the network configuration file to use
    network_mod_params : dict[str, Any] | None, default = None
        Keyword arguments for the network modification function
    hyperparams : dict[str, Any] | None, default = None
        Optional hyperparameters for the test
    network_mod_fn : Callable[[dict[str, dict[str, Any] | list[dict[str, Any]]], Any], None] | None, default = None
        Function to modify the network configuration before training
    custom_fn : Callable[[Any], None] | None, default = None
        Function to run custom logic after training, receives locals() as argument
    """
    name: str
    description: str
    dataset_args: dict[str, Any]
    net_name: str
    network_mod_params: dict[str, Any] | None = None
    hyperparams: dict[str, Any] | None = None
    network_mod_fn: Callable[
                        [dict[str, dict[str, Any] | list[dict[str, Any]]], Any],
                        None,
                    ] | None = None
    custom_fn: Callable[[Any], None] | None = None


def mod_network(nets_dir: str, test: TestConfig) -> str:
    """
    Modify the network JSON file according to the test configuration and save as a temp file.

    Parameters
    ----------
    nets_dir : str
        Path to the network configurations directory
    test : TestConfig
        The test configuration containing the network modification parameters

    Returns
    -------
    str
        Path to the new temporary JSON file with updated latent dimension
    """
    new_name: str
    config: dict[str, dict[str, Any] | list[dict[str, Any]]]
    file: TextIO

    if '.json' in test.net_name:
        test.net_name.replace('.json', '')

    if not test.network_mod_fn:
        return test.net_name

    new_name = test.net_name + '_temp'

    with open(os.path.join(ROOT, nets_dir, test.net_name) + '.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    test.network_mod_fn(config, **test.network_mod_params or {})

    with open(os.path.join(ROOT, nets_dir, new_name) + '.json', 'w', encoding='utf-8') as file:
        json.dump(config, file)

    return new_name


def run_tests(
        save: int | str,
        load: int | str,
        results_path: str,
        tests: list[TestConfig],
        config: str | dict[str, Any] = '../config.yaml') -> None:
    """
    Run a series of tests on the neural network using the provided configurations.

    Parameters
    ----------
    save : int | str
        File name to save the results, or 0 to not save
    load : int | str
        File name to load the networks from, or 0 to not load
    results_path : str
        Path to save the results DataFrame
    tests : list[TestConfig]
        Test configurations
    config : str | dict[str, Any], default = '../config.yaml'
        Configuration file path or dictionary
    """
    val_frac: float
    nets_dir: str
    net_name: str
    loaders: tuple[DataLoader, ...]
    dataset_args: dict[str, Any] = {}
    results: pd.DataFrame = pd.DataFrame([], columns=[
        'net_path',
        'description',
        'losses',
        'test_config',
        *(tests[0].dataset_args if tests[0].dataset_args else []),
        *(tests[0].network_mod_params if tests[0].network_mod_params else []),
        *(tests[0].hyperparams if tests[0].hyperparams else []),
    ])
    net: BaseNetwork
    dataset: BaseDataset
    test: TestConfig

    if isinstance(config, str):
        _, config = open_config('main', config)

    nets_dir = str(os.path.join(ROOT, config['data']['network-configs-directory']))

    for test in tests:
        print(f'Running test: {test.description}', flush=True)

        # 1. Prepare network JSON
        net_name = mod_network(nets_dir, test)

        # 2. Prepare config
        config['training'].update(test.hyperparams or {})
        config['training']['network-name'] = net_name
        config['training']['network-save'] = f'{save}.{test.name}' if save else 0
        config['training']['network-load'] = f'{load}.{test.name}'
        config['training']['description'] = test.description
        val_frac = config['training']['validation-fraction']

        if not load or not os.path.exists(save_name(
            config['training']['network-load'],
            str(os.path.join(ROOT, config['output']['network-states-directory'])),
            config['training']['network-name'],
        )):
            config['training']['network-load'] = 0

        # 3. Prepare or reuse dataset
        if not test.dataset_args == dataset_args:
            dataset = DarkDataset(**test.dataset_args)
            dataset_args = test.dataset_args

        # 4. Continue as before, using the cached dataset
        net = net_init(dataset, config=config)
        loaders = loader_init(
            dataset,
            batch_size=config['training']['batch-size'],
            ratios=(1 - val_frac, val_frac) if net.idxs is None else (1,),
            idxs=None if net.idxs is None else dataset.idxs[np.isin(
                dataset.extra['ids'],
                net.idxs,
            )],
        )
        net.idxs = dataset.extra['ids'].iloc[loaders[0].dataset.indices] if net.idxs is None else \
            net.idxs

        # 4. Custom user logic (optional)
        if test.custom_fn:
            test.custom_fn(locals())

        # 5. Train and evaluate
        net.training(config['training']['epochs'], loaders)
        net.save()
        dataset.high_dim = net.transforms['inputs'](dataset.high_dim, back=True)
        dataset.low_dim = net.transforms['targets'](dataset.low_dim, back=True)
        results = pd.concat((results, pd.DataFrame([{
            'net_path': net.save_path,
            'description': test.description,
            'losses': np.array(net.losses),
            'network_mod_fn': test.network_mod_fn.__name__ if test.network_mod_fn else None,
            'custom_fn': test.custom_fn.__name__ if test.custom_fn else None,
            **(test.dataset_args or {}),
            **(test.hyperparams or {}),
            **(test.network_mod_params or {}),
        }])))

        if test.network_mod_fn:
            os.remove(os.path.join(nets_dir, net_name) + '.json')

        if not save:
            continue

        # 6. Save results
        with open(os.path.join(ROOT, results_path), 'wb') as file:
            pickle.dump(results, file)
