"""
Runs several Gaussian training cycles with different cross-sections
"""
import os
import pickle
from typing import Any

import numpy as np
import netloader.networks as nets
from numpy import ndarray, floating
from torch.utils.data import DataLoader

# from src.utils import plots
from src import plots
from src.main import net_init
from src.utils.utils import open_config
from src.utils.data import GaussianDataset, loader_init


def init(
        data_path: str,
        known: list[float],
        unknown: list[float],
        config: str | dict[str, Any] = '../config.yaml') -> tuple[
        tuple[DataLoader, DataLoader],
        nets.BaseNetwork,
        GaussianDataset]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    data_path : str
        Path to the Gaussian dataset
    known : list[float]
        Known classes and labels
    unknown : list[float]
        Known classes with unknown labels to pass to the Gaussian dataset
    config : str | dict[str, Any], default = '../config.yaml'
        Configuration dict or path to the configuration dict

    Returns
    -------
    tuple[tuple[Dataloader, Dataloader], BaseNetwork, NormFlow]
        Train & validation dataloaders, neural network and flow training objects
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Load config parameters
    batch_size = config['gaussian']['batch-size']
    val_frac = config['gaussian']['validation-fraction']

    # Fetch dataset
    dataset = GaussianDataset(data_path, known, unknown)

    net = net_init(dataset)
    loaders = loader_init(dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    net.idxs = dataset.idxs
    dataset.labels = net.header['targets'](dataset.labels, back=True)
    dataset.images = net.in_transform(dataset.images, back=True)
    return loaders, net, dataset


def gaussian_test(
        known: list[float],
        unknown: list[float],
        unseen: list[float],
        runs: int = 5,
        config: str | dict[str, Any] = '../config.yaml',
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Runs a test for a set of known, unknown, and unseen classes

    Parameters
    ----------
    known : list[float]
        List of known classes
    unknown : list[float]
        List of known classes with unknown labels
    unseen : list[float]
        List of unknown classes with unknown labels
    runs : int, default = 5
        Number of runs to average over
    config : str | dict[str, Any], default = '../config.yaml'
        Configuration dict or path to the configuration dict

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Medians, means, stds, and stes of the runs
    """
    i: int
    epochs: int
    stds: list[floating]
    stes: list[floating]
    means: list[floating]
    medians: list[floating]
    stds_: list[ndarray] = []
    stes_: list[ndarray] = []
    means_: list[ndarray] = []
    medians_: list[ndarray] = []
    loaders: tuple[DataLoader, DataLoader]
    loaders_unseen: tuple[DataLoader, DataLoader]
    data: dict[str, Any]
    idxs: ndarray
    dataset: GaussianDataset
    dataset_unseen: GaussianDataset
    net: nets.BaseNetwork

    if isinstance(config, str):
        config = open_config('gaussian', config)[1]

    # Initialise network
    epochs = config['gaussian']['epochs']
    loaders, net, dataset = init(config['data']['data-path'], known, unknown, config=config)
    loaders_unseen, _, dataset_unseen = init(
        config['data']['data-path'],
        unseen,
        [],
        config=config,
    )

    for i in range(runs):
        print(f'\nRun {i + 1}/{runs}')
        stds = []
        stes = []
        means = []
        medians = []
        net = net_init(dataset)
        dataset_unseen.images = net.in_transform(dataset_unseen.images)
        dataset_unseen.labels = net.header['targets'](dataset_unseen.labels)

        # Train network
        net.training(epochs, loaders)
        data = net.predict(loaders_unseen[1])
        data['latent'][:, 0] = net.header['targets'](data['latent'][:, 0], back=True)

        # Summaries for each class
        for class_ in np.unique(data['targets']):
            idxs = data['targets'].squeeze() == class_
            medians.append(np.median(data['latent'][idxs, 0]))
            means.append(np.mean(data['latent'][idxs, 0]))
            stds.append(np.std(data['latent'][idxs, 0]))
            # medians.append(np.median(data['preds'][idxs]))
            # means.append(np.mean(data['preds'][idxs]))
            # stds.append(np.std(data['preds'][idxs]))
            stes.append(stds[-1] / np.sqrt(np.count_nonzero(idxs)))

        # Summaries for each run
        stds_.append(np.stack(stds))
        stes_.append(np.stack(stes))
        means_.append(np.stack(means))
        medians_.append(np.stack(medians))

        # Untransform dataset back to default state
        dataset.images = net.in_transform(dataset.images, back=True)
        dataset.labels = net.header['targets'](dataset.labels, back=True)
        dataset_unseen.images = net.in_transform(dataset_unseen.images, back=True)
        dataset_unseen.labels = net.header['targets'](dataset_unseen.labels, back=True)

    return np.stack(medians_), np.stack(means_), np.stack(stds_), np.stack(stes_)


def main(config_path: str = '../config.yaml'):
    """
    Main function for testing Gaussian toy datasets

    Parameters
    ----------
    config_path : str, default = '../config.yaml'
        Path to the configuration file
    """
    start: int
    save_path: str
    known: list[float]
    unknown: list[float]
    data: dict[str, list[str] | list[float] | list[list[float]] | list[ndarray]]
    config: dict[str, Any] = open_config('gaussian', config_path)[1]
    meds: ndarray
    stds: ndarray
    stes: ndarray
    means: ndarray

    save_path = config['output']['tests-path']

    if config['gaussian']['clear'] and os.path.exists(save_path):
        os.remove(save_path)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = {
            'known': config['gaussian']['known'],
            'unknown': config['gaussian']['unknown'],
            'unseen': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.05],
            'meds': [],
            'means': [],
            'stds': [],
            'stes': [],
            'names': config['gaussian']['names'],
        }

    if len(data['unknown']) == 1:
        data['unknown'] *= len(data['known'])

    start = len(data['meds'])

    for known, unknown in zip(data['known'][start:], data['unknown'][start:]):
        means, meds, stds, stes = gaussian_test(known, unknown, data['unseen'], config=config)
        data['meds'].append(meds)
        data['means'].append(means)
        data['stds'].append(stds)
        data['stes'].append(stes)

        if save_path:
            with open(save_path, 'wb') as file:
                pickle.dump(data, file)

    plots.PlotGaussianPreds(
        np.mean(data['means'], axis=1),
        data['unseen'],
        labels=data['names'],
        uncertainties=np.std(data['means'], axis=1),
    )
    # accuracies = np.stack(accuracies)
    # pd.set_option('display.max_columns', 20)
    # pd.set_option('display.width', 500)
    # print(pd.DataFrame(
    #     (
    #         np.mean(means, axis=0),
    #         np.mean(stds, axis=0),
    #         np.mean(medians, axis=0),
    #         np.std(medians, axis=0)
    #     ),
    #     index=['Means', 'Stds', 'Medians', "Median's Std"],
    #     columns=np.unique(known + unseen_unknown),
    # ).round(3))


if __name__ == '__main__':
    main()
