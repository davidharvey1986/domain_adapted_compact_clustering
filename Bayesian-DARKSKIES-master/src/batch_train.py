"""
Trains a network multiple times for different datasets
"""
import os
import pickle
from typing import Any

import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader
from netloader.networks import BaseNetwork
from netloader.utils.utils import save_name

from src.utils import analysis
from src.main import init, net_init
from src.utils.data import DarkDataset
from src.utils.utils import open_config, ROOT


def update_sims(new_sims: list[str], current_sims: ndarray) -> ndarray:
    """
    Updates the current array of simulations, makes the array unique, and removes any sims with an
    exclamation

    Parameters
    ----------
    new_sims : list[str]
        New simulations to add
    current_sims : (N) ndarray
        N current simulations

    Returns
    -------
    (M) ndarray
        M simulations including the previous and new sims, minus any that need removing
    """
    current_sims = np.append(current_sims, new_sims)
    current_sims = np.unique(current_sims)

    if len(current_sims) == 0:
        return current_sims

    idxs = np.char.find(current_sims, '!') != -1
    current_sims = np.char.replace(current_sims, '!', '')
    return current_sims[~np.isin(current_sims, current_sims[idxs])]


def main(config_path: str = '../config.yaml'):
    """
    Main function for batch training of the network

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        Path to the configuration file
    """
    cumulative: bool
    unknown_cumulative: bool
    i: int
    j: int
    epochs: int
    repeats: int
    save_num: int
    key: str
    name: str
    net_name: str
    states_dir: str
    sims: list[str]
    unknown_sims: list[str]
    known: list[list[str]]
    unknown: list[list[str]]
    loaders: tuple[DataLoader, DataLoader]
    config: dict[str, Any]
    batch_config: dict[str, Any]
    results: dict[int, dict[str, str | list[ndarray] | ndarray] | BaseNetwork] = {}
    meds: ndarray
    stes: ndarray
    means: ndarray
    current_known: ndarray = np.array([])
    current_unknown: ndarray = np.array([])
    net: BaseNetwork
    dataset: DarkDataset

    _, batch_config = open_config('batch-train', config_path)
    _, config = open_config('main', config_path)
    _, batch_config['training'] = open_config(
        'training',
        str(os.path.join(batch_config['data']['config-dir'], batch_config['data']['config'])),
    )

    for key, value in batch_config.items():
        config[key] |= value

    cumulative = config['training']['cumulative']
    unknown_cumulative = config['training']['unknown-cumulative']
    epochs = config['training']['epochs']
    repeats = config['training']['repeats']
    save_num = config['training']['batch-save']
    load_num = config['training']['batch-load']
    net_name = config['training']['network-name']
    description = config['training']['description']
    names = config['training']['test-names']
    known = config['training']['known-simulations']
    unknown = config['training']['unknown-simulations']
    states_dir = str(os.path.join(ROOT, config['output']['network-states-directory']))

    if len(unknown) == 1:
        unknown *= len(known)

    if len(known) == 1:
        known *= len(unknown)

    if len(known) != len(unknown):
        raise ValueError(f'Number of known simulation tests ({len(known)}) does not equal the '
                         f'number of unknown simulation tests ({len(unknown)})')

    for i, (sims, unknown_sims, name) in enumerate(zip(known, unknown, names)):
        if cumulative:
            current_known = update_sims(sims, current_known)
        else:
            current_known = np.array(sims)

        if unknown_cumulative:
            current_unknown = update_sims(unknown_sims, current_unknown)
        else:
            current_unknown = np.array(unknown_sims)

        config['training']['description'] = f'{name}, {description}' if description else name
        results[i] = {
            'meds': [],
            'means': [],
            'stes': [],
            'log_meds': [],
            'log_means': [],
            'log_stes': [],
            'targets': [],
            'nets': [],
            'description': config['training']['description'],
            'sims': current_known.tolist(),
            'unknown_sims': current_unknown.tolist(),
        }

        # Initialise datasets
        if os.path.exists(save_name(
                f'{load_num}.{i}.0',
                states_dir,
                net_name,
        )) and load_num:
            config['training']['network-load'] = f'{load_num}.{i}.0'
        else:
            config['training']['network-load'] = 0

        config['training']['network-save'] = 0
        loaders, net, dataset = init(
            current_known.tolist(),
            config,
            unknown=current_unknown.tolist(),
        )

        for j in range(repeats):
            config['training']['network-save'] = f'{save_num}.{i}.{j}'

            if os.path.exists(save_name(
                    f'{load_num}.{i}.{j}',
                    states_dir,
                    net_name,
            )) and load_num:
                config['training']['network-load'] = f'{load_num}.{i}.{j}'
            else:
                config['training']['network-load'] = 0

            print(
                f"\nSave: {config['training']['network-save']}\n"
                f'Sims: {current_known.tolist()}\n'
                f'Unknown sims: {current_unknown.tolist()}\n'
                f"Name: {config['training']['description']}"
            )

            # Remove dataset transform & initialise network
            dataset.high_dim = net.transforms['inputs'](dataset.high_dim, back=True)
            dataset.low_dim = net.transforms['targets'](dataset.low_dim, back=True)
            net = net_init(dataset, config=config)
            net.idxs = dataset.extra['ids'].iloc[loaders[0].dataset.indices]
            net.description = config['training']['description']
            net.training(epochs, loaders)
            net.save()
            data = net.predict(loaders[1])
            data['targets'] = dataset.correct_unknowns(data['targets'].squeeze())
            data['targets'] = dataset.unique_labels(
                data['targets'],
                dataset.extra['sims'].iloc[data['ids'].astype(int)],
            )
            # data['latent'] = net.transforms['targets'](data['preds']).numpy() # For Encoders
            meds, means, stes = analysis.summary(data)[:3]

            results[i]['log_meds'].append(meds)
            results[i]['log_means'].append(means)
            results[i]['log_stes'].append(stes)

            meds = net.transforms['targets'](meds, back=True)
            means, stes = net.transforms['targets'](means, back=True, uncertainty=stes)

            results[i]['meds'].append(meds)
            results[i]['means'].append(means)
            results[i]['stes'].append(stes)
            results[i]['targets'].append(np.unique(data['targets']))
            results[i]['nets'].append(net.save_path)

        for key in ('meds', 'means', 'stes', 'log_meds', 'log_means', 'log_stes', 'targets'):
            results[i][key] = np.stack(results[i][key])

    with open(os.path.join(
        ROOT,
        config['output']['batch-train-dir'],
        f'batch_train_{save_num}.pkl',
    ), 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    main()
