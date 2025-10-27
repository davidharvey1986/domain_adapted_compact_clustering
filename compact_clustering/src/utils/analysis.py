"""
Functions to analyse the neural network
"""
import os
import pickle
from typing import Any, BinaryIO

import torch
import numpy as np
from numpy import ndarray
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
from netloader.data import BaseDataset, loader_init


def _key_summary(
        key: str,
        data: dict[str, ndarray]) -> dict[str, ndarray]:
    """
    Gets the mean, standard deviation, standard errors, and errors for a given key

    Parameters
    ----------
    key : str
        Key to get the statistics for
    data : dict[str, (N,M) ndarray]
        N repeats to get the statistics for for M parameters

    Returns
    -------
    dict[str, (M) ndarray]
        Statistics for the given key
    """
    post_data: dict[str, ndarray] = {
        key: np.mean(data[key], axis=0),
        f'{key}_stds': np.std(data[key], axis=0, ddof=1),
        f'{key}_errors': np.sqrt(np.sum(
            data['log_stes' if 'log' in key else 'stes'] ** -2,
            axis=0,
        ) ** -1),
    }
    post_data[f'{key}_stes'] = post_data[f'{key}_stds'] / np.sqrt(len(data[key]))

    if 'med' not in key:
        post_data[f'{key}_weighted'] = np.average(
            data[key],
            weights=data['log_stes' if 'log' in key else 'stes'] ** -2,
            axis=0,
        )
    return post_data


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    """
    Generates a summary of results for a single run from a batch training run

    Parameters
    ----------
    data : dict[str, Any]
        Data from one run from a batch training run

    Returns
    -------
    dict[str, Any]
        Summary of data from one run from a batch training run
    """
    key: str
    post_data: dict[str, str | ndarray] = {}
    primary_keys: ndarray = np.char.add(
        np.array(['', 'log_'])[:, np.newaxis],
        ['means', 'meds'],
    ).flatten()

    for key in primary_keys:
        post_data |= _key_summary(str(key), data)

    for key in set(data.keys()) - set(primary_keys):
        post_data[key] = data[key]

    return post_data


def _batch_train_summary(
        num: int,
        dir_: str,
        idx: int | None = None) -> dict[str, list[Any]]:
    """
    Generates a summary for a single file generated from batch_train.py

    Parameters
    ----------
    num : int
        Batch train file number
    dir_ : str
        Path to the directory for the batch training data
    idx : int | None, default = None
        Which run to get the results for, if None, all runs will be used

    Returns
    -------
    dict[str, list[Any]]
        Summary of the batch training data
    """
    run: int
    key: str
    data: dict[int, dict[str, Any]]
    post_data: dict[str, list[Any]] = {}
    datum: dict[str, Any]
    val: Any

    with open(os.path.join(dir_, f'batch_train_{num}.pkl'), 'rb') as file:
        data = pickle.load(file)

    # Loop through all runs in the batch training file
    for run, datum in data.items():
        if idx is not None and run != list(data.keys())[idx]:
            continue

        datum = _summary(datum)

        # Add run summary to batch train summary
        for key, val in datum.items():
            if isinstance(val, ndarray):
                val = val.tolist()

            if key in post_data:
                post_data[key].append(val)
            else:
                post_data[key] = [val]
    return post_data


def _red_chi_acc(
        dof: int,
        values: ndarray,
        target: ndarray,
        errors: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Calculates the reduced chi square and mean squared error along the last dimension

    Parameters
    ----------
    values : (...,L) ndarray
        L predicted values
    target : (...,L) ndarray
        L target values
    errors : (...,L) ndarray
        L uncertainties

    Returns
    -------
    tuple[(...) ndarray, (...) ndarray, (...) ndarray, (...) ndarray]
        Reduced chi square, reduced chi square uncertainty, mean squared error, and mean squared
        error uncertainty
    """
    red_chi = np.sum(((values - target) / errors) ** 2, axis=-1) / dof
    red_chi_error = 2 * np.sqrt(red_chi / dof)
    acc = np.mean((values - target) ** 2, axis=-1)
    acc_error = 2 * np.sqrt(np.sum(((values - target) * errors) ** 2, axis=-1)) / values.shape[-1]
    return red_chi, red_chi_error, acc, acc_error


def batch_train_summary(
        num: int | range,
        dir_: str,
        idx: int | None = None) -> dict[str, list[list[Any]] | ndarray]:
    """
    Calculate a summary for data generated from batch_train.py

    Parameters
    ----------
    num : int | range
        Batch train number or range of numbers
    dir_ : str
        Path to the directory for the batch training data
    idx : int | None, default = None
        Which run to get the results for, if None, all runs will be used

    Returns
    -------
    dict[str, list[list[Any]] | ndarray]
        Summary of data generated from batch_train.py, will try to form an array, but will fall back
        on list for all dimensions from the first dimension that can no longer be an array, so the
        shape will be (B,R,N,...) where B is the number of batch train files, R is the number of
        runs in each file, and N is the number of repeats for each run
    """
    i: int
    key: str
    val: list[Any]
    post_data: dict[str, list[list[Any]] | ndarray] = {}
    data: dict[str, list[Any]]
    num = range(num, num + 1) if isinstance(num, int) else num

    for i in num:
        data = _batch_train_summary(i, dir_, idx)

        for key, val in data.items():
            if isinstance(val, ndarray):
                val = val.tolist()

            if key in post_data:
                post_data[key].append(val)
            else:
                post_data[key] = [val]

    for key, val in post_data.items():
        try:
            post_data[key] = np.array(val)
        except ValueError:
            try:
                post_data[key] = np.array(val, dtype=object)
            except ValueError:
                pass
    return post_data


def distribution_func(
        data: ndarray,
        norm: bool = False,
        cumulative: bool = False,
        bins: int = 500,
        range_: tuple[float, float] | None = None,
        grid: ndarray | None = None) -> tuple[ndarray, ndarray]:
    """
    Calculates the x and y-values, PDF, or CDF for a distribution of points using a Gaussian kernel

    Parameters
    ----------
    data : ndarray
        Distribution to get the PDF with shape (N) and type float, where N is the number of points
    norm : bool, default = False
        If the distribution should be normalised to one
    cumulative : bool, default = False
        If the distribution should be cumulatively summed
    bins : int, default = 500
        Number of bins to use for the Gaussian kernel
    range_ : tuple[float, float] | None, default = None
        Range to get the PDF for, used if grid is None
    grid : ndarray | None, default = None
        Grid of values to get the PDF for with shape (N) and type float, range_ is required if grid
        is None

    Returns
    -------
    tuple[ndarray, ndarray]
        Grid of values with shape (N) and type float, and PDF of shape (N) and type float
    """
    distribution: ndarray
    kernel: gaussian_kde = gaussian_kde(data)

    if grid is None and range_ is not None:
        grid = np.mgrid[range_[0]:range_[1]:bins * 1j]
    if grid is None and range_ is None:
        raise ValueError('Either argument range_ or grid must be specified')

    distribution = kernel(grid)

    if norm:
        distribution /= np.sum(distribution)

    if cumulative:
        return grid, np.cumsum(distribution)
    return grid, distribution


def distributions(data: ndarray, targets: ndarray) -> ndarray:
    """
    Splits data from different classes into distributions for each class

    Parameters
    ----------
    data : ndarray
        Data values with shape (...,N), where N is the number of points
    targets : ndarray
        Target values that belong to C classes for each point with shape (...,N), where N is the
        number of points

    Returns
    -------
    ndarray
        Distributions for each class with shape (...,C), where C is the number of classes and each
        element is a list of values
    """
    shape: tuple[int, ...]
    idxs: ndarray
    new_data: ndarray = np.empty(
        [*data.shape[:-1], len(np.unique(targets[0, 0]))],
        dtype=object,
    )

    for shape in np.ndindex(new_data.shape):
        idxs = targets[*shape[:-1]] == np.unique(targets[*shape[:-1]])[shape[-1]]
        new_data[*shape] = data[*shape[:-1], idxs]

    try:
        return np.array(new_data.tolist())
    except ValueError:
        return new_data


def gen_predictions(
        batch_size: int,
        val_frac: float,
        nets: ndarray,
        dataset: BaseDataset,
        idxs: tuple[ndarray, ...] | list[ndarray] | ndarray | None = None) -> ndarray:
    """
    Generates predictions for each network

    Parameters
    ----------
    batch_size : int
        Batch size for the data loader
    val_frac : float
        Validation fraction for the data loader
    nets : ndarray
        Networks to generate predictions for of shape (...,R) and type BaseNetwork, where R is the
        number of repeats in each test
    dataset : BaseDataset
        Dataset to generate predictions for
    idxs : tuple[ndarray, ...] | list[ndarray] | ndarray | None, default = None
        Dataset indexes for creating the subsets with shape (N,S), where N is the number of subsets
        and S is the number of samples in each subset, will override net indexes if provided

    Returns
    -------
    ndarray
        Predictions for each network of shape (...,R) and type float
    """
    shape: tuple[int, ...]
    loaders: tuple[DataLoader, ...]
    low_dim: ndarray = dataset.low_dim.copy()
    high_dim: ndarray = dataset.high_dim.copy()
    predictions: ndarray = np.empty(nets.shape, dtype=object)
    loader: DataLoader

    for shape in np.ndindex(predictions.shape):
        # Create data loader only for one network in repeats
        if shape[-1] == 0:
            nets[*shape].transforms['inputs'][1]._shape = high_dim.shape[1:]
            dataset.high_dim = nets[*shape].transforms['inputs'](high_dim.copy())
            dataset.low_dim = nets[*shape].transforms['targets'](low_dim.copy())

            loaders = loader_init(
                dataset,
                batch_size=batch_size,
                ratios=(1 - val_frac, val_frac) if idxs is None and nets[*shape].idxs is None else
                (1,),
                idxs=dataset.idxs[np.isin(
                    dataset.extra['ids'],
                    nets[*shape].idxs,
                )] if idxs is None and nets[*shape].idxs is not None else idxs,
            )
            print(f'Loader Dataset Lengths: {[len(loader.dataset) for loader in loaders]}')

        # Generate predictions
        if predictions[*shape] is None:
            nets[*shape].to('cuda')
            predictions[*shape] = nets[*shape].predict(loaders[1])
            nets[*shape].to('cpu')
            torch.cuda.empty_cache()

    dataset.low_dim = low_dim
    dataset.high_dim = high_dim
    return predictions


def hyperparam_summary(path: str) -> tuple[list[int], ndarray, ndarray]:
    """
    Returns the accuracy and latent loss with standard deviations from the hyperparam_search results

    Parameters
    ----------
    path : str
        Path to the hyperparameter search results

    Returns
    -------
    tuple[list[int], (Nd,Ns,2) ndarray, (Nd,Ns,2) ndarray]
        List of latent dimensions, mean accuracy and latent loss, and standard deviations for Nd
        latent dimension tests and Ns simulation tests
    """
    i: int
    key: str
    latent_dims: list[int]
    loss: list[tuple[float, float]]
    list_dict: list[dict[str, Any]] = []
    mean_losses: list[ndarray] | ndarray = []
    std_losses: list[ndarray] | ndarray = []
    value: dict[str, Any]
    data: dict[str, Any]
    file: BinaryIO

    with open(path, 'rb') as file:
        data = pickle.load(file)

    for value in data.values():
        list_dict.append(value)

    data = {}

    for key in list_dict[0]:
        data[key] = []

    for value in list_dict:
        for key in data:
            data[key].append(value[key])

    data['mean_loss'] = []
    data['std_loss'] = []

    for loss in data['losses']:
        data['mean_loss'].append(np.mean(loss, axis=0))
        data['std_loss'].append(np.std(loss, axis=0))

    for i in range(0, len(data['mean_loss']), 8):
        mean_losses.append(data['mean_loss'][i:i + 8])
        std_losses.append(data['std_loss'][i:i + 8])

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)
    latent_dims = data['latent_dim'][:8]
    return latent_dims, mean_losses, std_losses


def mult_distributions(data: ndarray, bins: int = 500) -> tuple[ndarray, ndarray]:
    """
    Multiplies multiple distributions into a single distribution using a Gaussian kernel

    Parameters
    ----------
    data : ndarray
        Distributions to multiply with shape (...,R,C,N) and type float, where C is the number of
        classes, R is the number of distributions, and N is the number of points, or shape (...,R,C)
        and type object, where each element is a list of points
    bins : int, default = 500
        Number of bins when approximating the distributions using a gaussian kernel

    Returns
    -------
    tuple[ndarray, ndarray]
        Grid with shape (...,B) and type float, where B is the number of bins, and new
        distributions with shape (...,C,B) and type float
    """
    shape: tuple[int, ...]
    new_data: ndarray
    grids: ndarray = np.empty(
        (*(data.shape[:-2] if data.dtype.type == np.object_ else data.shape[:-3]), bins),
    )

    # Change data from (...,R,C,N) to (...,C,R,N)
    data = data.swapaxes(-1 if data.dtype.type == np.object_ else -3, -2)
    new_data = np.empty((
        *data.shape[:-1 if data.dtype.type == np.object_ else -2],
        bins,
    ))

    # Calculate the grid for each set of (R,C,N)
    for shape in np.ndindex(grids.shape[:-1]):
        grids[*shape] = np.mgrid[
                        min(np.min(values) for values in data[*shape].reshape(
                            *((-1,) if data.dtype.type == np.object_ else (-1, data.shape[-1])),
                        )):max(np.max(values) for values in data[*shape].reshape(
                            *((-1,) if data.dtype.type == np.object_ else (-1, data.shape[-1])),
                        )):bins * 1j
                        ]

    for shape in np.ndindex(data.shape[slice(None if data.dtype.type == np.object_ else -1)]):
        # If first distribution for the given class
        if shape[-1] == 0:
            new_data[*shape[:-1]] = distribution_func(
                data[*shape],
                bins=bins,
                grid=grids[*shape[:-2]],
            )[1]
        # All other distributions for the given class are multiplied with the current distribution
        else:
            new_data[*shape[:-1]] *= distribution_func(
                data[*shape],
                bins=bins,
                grid=grids[*shape[:-2]],
            )[1] + 1e-6
    return grids, new_data


def phys_params(
        data: dict[str, ndarray],
        names: ndarray,
        stellar_frac: ndarray,
        mass: ndarray) -> tuple[list[ndarray], list[ndarray]]:
    """
    Gets several physical parameters for the predicted data and creates a list of the data for each
    simulation

    Parameters
    ----------
    data : dict[str, (M,...)]
        M predicted data
    names : (N) ndarray
        N names for the dataset
    stellar_frac : (N) ndarray
        N stellar fractions for the dataset
    mass : (N) ndarray
        N masses for the dataset

    Returns
    -------
    tuple[list[(B,Z) ndarray], list[(B,6) ndarray]]
        B latent space values of dimensions Z for each simulation and the corresponding B physical
        parameters for each simulation
    """
    value: float
    sim: str
    key: str
    latents: list[ndarray] = []
    params: list[ndarray] = []
    sim_delta_tk: dict[str, float] = {
        'darkskies': 8,
        'hi': 8.2,
        'low': 7.8,
        'bahamas': 8,
        'flamingo': 8.07,
        'tng': 8,
    }
    sim_dm_mass: dict[str, float] = {
        'darkskies': 6.9e7,
        'bahamas': 5.5e9,
        'flamingo': 7.1e8,
        'tng': 6.1e7,
    }
    sim_b_mass: dict[str, float] = {
        'darkskies': 1.1e9,
        'bahamas': 1.1e9,
        'flamingo': 1.3e8,
        'tng': 1.2e7,
    }
    idxs: ndarray
    names = names[data['ids'].astype(int)]
    stellar_frac = stellar_frac[data['ids'].astype(int)]
    mass = mass[data['ids'].astype(int)]

    for sim in names[np.unique(data['targets'], return_index=True)[1]]:
        idxs = sim == names

        for key, value in sim_delta_tk.items():
            if key in sim.lower():
                delta_tk = np.ones(np.count_nonzero(idxs)) * value

        for key, value in sim_dm_mass.items():
            if key in sim.lower():
                dm_mass = np.ones(np.count_nonzero(idxs)) * np.log10(value)

        for key, value in sim_b_mass.items():
            if key in sim.lower():
                b_mass = np.ones(np.count_nonzero(idxs)) * np.log10(value)

        latents.append(data['latent'][idxs])
        params.append(np.stack((
            np.log10(data['targets'][idxs]),
            np.log10(mass[idxs]),
            stellar_frac[idxs],
            delta_tk,
            dm_mass,
            b_mass,
        ), axis=-1))

    return latents, params


def pred_distributions(targets: ndarray, preds: ndarray) -> list[ndarray]:
    """
    Generate the predicted distribution for each simulation

    Parameters
    ----------
    targets : (N) ndarray
        N target values for the predicted data
    preds : (N,...) ndarray
        N predicted data

    Returns
    -------
    list[ndarray]
        Predicted distribution for each simulation
    """
    data: list[ndarray] = []

    for target in np.unique(targets):
        data.append(preds[targets == target])

    return data


def probs_distributions(
        quantile_values: list[float],
        predictions: ndarray,
        nets: ndarray,
        dataset: BaseDataset,
        bins: int = 500) -> tuple[
    dict[str, ndarray],
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray]:
    """
    Calculates the probability distributions and quantiles for the given predictions from batch
    training and quantile values.

    Parameters
    ----------
    quantile_values : list[float]
        Quantiles to get the corresponding values for
    predictions : ndarray
        Predictions from each network in batch training of shape (...,R) and type object, where R is
        the number of repeated networks
    nets : ndarray
        Network objects for each network prediction of shape (...,R) and type object
    dataset : BaseDataset
        Dataset that contains the labels for calculating the distributions for each label class
    bins : int, default = 500
        Number of bins for the product of distributions

    Returns
    -------
    tuple[dict[str, ndarray], ndarray, ndarray, ndarray, ndarray, ndarray]
        Dictionary of predictions for each network with predictions of shape (...,R) and type
        object; distributions for each class for each network of shape (...,R,C) and type float,
        where C is the number of classes; distribution products of shape (...,C,B) and type float,
        where B is the number of bins uniformly sampled from the PDF; x-values for distribution
        products of shape (...,B) and type float; distribution products normalised into a PDF of
        shape (...,C,B) and type float; quantiles of the PDF of shape (Q,...,C) and type float,
        where Q is the number of quantiles
    """
    i: int
    percentile: float
    key: str
    shape: tuple[int, ...]
    data_pred: dict[str, ndarray] = dict(zip(
        predictions.flatten()[0].keys(),
        [np.empty_like(predictions) for _ in range(len(predictions.flatten()[0].keys()))],
    ))
    dists: ndarray
    grids: ndarray
    probs: ndarray
    cumsums: ndarray
    quantiles: ndarray
    new_distributions: ndarray

    # Convert ndarray of dictionaries to dictionary of ndarrays
    for key in data_pred:
        for shape in np.ndindex(predictions.shape):
            data_pred[key][*shape] = predictions[*shape][key].copy()

    for key in data_pred:
        try:
            data_pred[key] = np.array(data_pred[key].tolist())
        except ValueError:
            pass

    # Conversion of Encoder to ClusterEncoder predictions
    if 'latent' not in data_pred:
        for shape in np.ndindex(data_pred['preds'].shape[:2]):
            data_pred['preds'][*shape] = nets[*shape, 0].transforms['targets'][1:](
                data_pred['preds'][*shape],
            )

    # Generate distributions for each set of predictions & product of repeats of distributions
    data_pred['targets'] = data_pred['targets'].squeeze(axis=-1)
    dists = distributions(
        data_pred['latent'][..., 0] if 'latent' in data_pred else
        data_pred['preds'].squeeze(axis=-1),
        dataset.low_dim[data_pred['ids'].astype(int)].squeeze(axis=-1),
    )
    grids, new_distributions = mult_distributions(dists, bins=bins)

    # Untransform grids
    for shape in np.ndindex(grids.shape[:-1]):
        grids[*shape] = nets[*shape, 0].transforms['targets'](grids[*shape], back=True)

    # Calculate quantile values
    probs = np.empty_like(new_distributions)
    cumsums = probs.copy()
    quantiles = np.empty((len(quantile_values), *cumsums.shape[:-1]))

    for shape in np.ndindex(new_distributions.shape[:-1]):
        probs[*shape] = new_distributions[*shape] / np.trapezoid(
            new_distributions[*shape],
            np.log10(grids[*shape[:-1]]),
        )
        cumsums[*shape] = np.cumsum(probs[*shape] / np.sum(probs[*shape]))

        for i, percentile in enumerate(quantile_values):
            quantiles[i, *shape] = grids[*shape[:-1], np.argmin(
                np.abs(cumsums[*shape] - percentile),
                axis=-1,
            )]
    return data_pred, dists, new_distributions, grids, probs, quantiles


def profiles(
        images: ndarray,
        norms: ndarray,
        names: ndarray,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Generates the total, X-ray fraction, and stellar fraction profiles for each simulation

    Parameters
    ----------
    images : (N,3,H,W) ndarray
        N images of height H and width W with channels total mass, X-ray, and stellar
    norms : (N,3) ndarray
        Normalisation factors for each channel
    names : (N) ndarray
        Simulation name for each image

    Returns
    -------
    tuple[(M) ndarray, (L) ndarray, (M,L) ndarray, (M,L) ndarray, (M,L) ndarray]
        M unique simulation names, radii, and total mass, X-ray fraction, and stellar fraction
        profiles with L bins which is equal to int(min(W,H)/4-0.5)
    """
    i: int
    j: int
    radius: int
    name: str
    centers: tuple[int, int]
    idxs: ndarray
    mask: ndarray
    total: ndarray
    x_ray: ndarray
    stellar: ndarray

    images *= norms.reshape(*norms.shape, *[1] * (len(images.shape) - len(norms.shape)))

    centers = (images.shape[-2] // 2, images.shape[-1] // 2)
    total = np.empty((len(np.unique(names)), int(np.ceil((min(centers) - 2) / 2))))
    x_ray = np.empty((len(np.unique(names)), int(np.ceil((min(centers) - 2) / 2))))
    stellar = np.empty((len(np.unique(names)), int(np.ceil((min(centers) - 2) / 2))))

    for i, name in enumerate(np.unique(names)):
        idxs = name == names

        for radius in range(2, min(centers), 2):
            j = (radius - 2) // 2
            mask = np.where(np.sqrt(np.add(*[array ** 2 for array in np.meshgrid(
                np.arange(images.shape[-2]) - images.shape[-2] // 2 + 0.5,
                np.arange(images.shape[-1]) - images.shape[-1] // 2 + 0.5,
            )])) < radius, 1, 0)
            total[i, j] = np.mean(
                np.sum(images[idxs, 0] * mask, axis=(-2, -1)) / (4 * np.pi * radius ** 2),
            )
            x_ray[i, j] = np.mean(
                np.sum(images[idxs, 1] * mask, axis=(-2, -1)) /
                np.sum(images[idxs, 0] * mask, axis=(-2, -1)),
            )
            stellar[i, j] = np.mean(
                np.sum(images[idxs, -1] * mask, axis=(-2, -1)) /
                np.sum(images[idxs, 0] * mask, axis=(-2, -1)),
            )

    return np.unique(names), np.arange(2, min(centers), 2) * 20, total, x_ray, stellar


def proj_1d(
        target_class: int | float | str,
        centers: ndarray,
        rel_vecs: ndarray,
        classes: ndarray) -> ndarray:
    """
    Projects a set of vectors belonging to the target class onto the direction of the centers of all
    other classes and the set of vectors belonging to different classes onto the direction between
    the centers of each class and the target class.

    Parameters
    ----------
    target_class : int | float | str
        Target class label to get the directional vectors to the center of all other classes
    centers : ndarray
        Global vectors pointing to the center of each class with shape (C,...) and type float,
        where C is the number of classes
    rel_vecs : ndarray
        Vectors relative to their respective class center with shape(C) and type object, where each
        element contains the vectors for each class of shape (N,...), where N is the number of
        vectors per class
    classes : ndarray
        Unique classes with the same order as centers and rel_vecs with shape (C) and type int |
        float | str matching target_class

    Returns
    -------
    ndarray
        Projected vectors with shape (2,C-1) and type object, where the first row is the projected
        vectors of the target class in the direction to all other classes, and the second row is the
        projected vectors of the other classes in the direction of the target class, where each
        element contains an array of shape (N) and type float
    """
    i: int
    vecs: ndarray
    idxs: ndarray = np.array(classes == target_class)
    direcs: ndarray = centers[~idxs] - centers[idxs]
    norms: ndarray = np.linalg.norm(direcs, axis=1)
    proj_vecs: ndarray = np.empty((2, len(idxs) - 1), dtype=object)

    # Project vectors of target class in direction of all other classes
    proj_vecs[0] = [*(rel_vecs[idxs][0] @ direcs.T / norms).swapaxes(0, 1)]

    # Project vectors of other classes in direction of target class
    for i, vecs in enumerate(rel_vecs[~idxs]):
        proj_vecs[1, i] = norms[i] + vecs @ direcs[i] / norms[i]
    return proj_vecs


def proj_all_inter_1d(vecs: ndarray, classes: ndarray) -> ndarray:
    """
    Projects all vectors belonging to each class in the direction of the center of their class to
    the centers of all other classes.

    Parameters
    ----------
    vecs : ndarray
        Global vectors with shape (N,...) and type float, where N is the total number of vectors
    classes : ndarray
        Class for each vector with shape (N)

    Returns
    -------
    ndarray
        Projected vectors with shape (C,2,C-1) and type object, where C is the number of classes,
        each row corresponds to each class being the target class that all vectors are projected in
        the direction of, the first column represents the vectors from the target class in the
        direction of all other classes, and the second column represents the vectors from all other
        classes in the direction of the target class, where each element contains an array of shape
        (N) and type float
    """
    class_: int | float | str
    proj_vecs: list[ndarray] = []
    centers: ndarray
    rel_vecs: ndarray

    centers, rel_vecs = relative_vecs(vecs, classes)

    for class_ in np.unique(classes):
        proj_vecs.append(proj_1d(class_, centers, rel_vecs, np.unique(classes)))

    return np.array(proj_vecs, dtype=object)


def relative_vecs(vecs: ndarray, classes: ndarray) -> tuple[ndarray, ndarray]:
    """
    Calculates a set of vectors relative to the center of the set of vectors for each class.

    Parameters
    ----------
    vecs : ndarray
        Vectors with shape (N,...) and type float, where N is the number of vectors
    classes : ndarray
        Class labels with shape (N)

    Returns
    -------
    tuple[ndarray, ndarray]
        Class centers with shape (C,...), where C is the number of classes; and relative vectors
        with shape (C,M,...) and type float or (C) with type object, where M is the number of
        vectors per class if there are an equal amount; otherwise, each element contains the vectors
        for each class of shape (M,...)
    """
    class_: int | float | str
    centers: list[ndarray] = []
    rel_vecs: list[ndarray] = []
    idxs: ndarray

    for class_ in np.unique(classes):
        idxs = np.array(classes == class_)
        centers.append(np.mean(vecs[idxs], axis=0))
        rel_vecs.append(vecs[idxs] - centers[-1])

    try:
        return np.array(centers), np.array(rel_vecs)
    except ValueError:
        return np.array(centers), np.array(rel_vecs, dtype=object)


def summary(data: dict[str, ndarray]) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Generates summary stats for the trained network

    Parameters
    ----------
    data : dictionary
        Data returned from the trained network

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        Medians, means, standard errors and accuracies for the network predictions
    """
    accuracies: list[float] = []
    stes: list[np.floating] = []
    means: list[np.floating] = []
    medians: list[np.floating] = []
    class_: float
    idxs: ndarray

    for class_ in np.unique(data['targets']):
        idxs = np.array(data['targets'] == class_)
        medians.append(np.median(data['latent'][idxs, 0]))
        means.append(np.mean(data['latent'][idxs, 0]))
        stes.append(np.std(data['latent'][idxs, 0]) / np.sqrt(np.count_nonzero(idxs)))
        accuracies.append(np.count_nonzero(data['preds'][idxs] == class_) / len(idxs))

    return np.stack(medians), np.stack(means), np.stack(stes), np.stack(accuracies)
