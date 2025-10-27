"""
Takes all pickle files in a directory and subdirectories of labels and images and combines them into
a single pickle file
"""
import os
import pickle
from typing import BinaryIO
from logging import getLogger

import numpy as np
from numpy import ndarray

# Required to initialise the logger
import src  # pylint: disable=unused-import


def load_pickle(path: str) -> tuple[
    dict[str, float | list[float] | list[ndarray]],
    list[ndarray] | ndarray
]:
    """
    Loads labels and images from pickle file and finds which is the labels and which is the images

    Parameters
    ----------
    path : str
        Path to the pickle file

    Returns
    -------
    tuple[dict[str, list[float] | list[ndarray]], list[ndarray] | ndarray]
        Labels and images
    """
    data: (tuple[dict[str, float | list[float] | list[ndarray]], list[ndarray] | ndarray] |
           tuple[list[ndarray] | ndarray, dict[str, list[float] | list[ndarray]]])
    file: BinaryIO

    with open(path, 'rb') as file:
        data = pickle.load(file)

    if isinstance(data[0], dict):
        return data

    return data[1], data[0]


def dict_list_append(
        dict1: dict[str, list[float | None] | list[ndarray]],
        dict2: dict[str, float | list[float] | list[ndarray] | ndarray],
) -> dict[str, list[float | None] | list[ndarray]]:
    """
    Merges two dictionaries of lists

    Parameters
    ----------
    dict1 : dict[str, list[float | None] | list[ndarray]]
        Primary dict to merge secondary dict into, can be empty
    dict2 : dict[str, float | list[float] | list[ndarray] | ndarray]
        Secondary dict to merge into primary dict, requires at least one element

    Returns
    -------
    dict[str, list[float | None] | list[ndarray]]
        First dict with second dict merged into it
    """
    dict1_len: int = 0
    dict2_len: int = 1
    key: str

    # If primary dict is not empty, find the length of a list in the dictionary
    if len(dict1.keys()) > 0:
        dict1_len = len(dict1[list(dict1.keys())[0]])

    # If the secondary dict contains a list of items, find the length of the lists
    if isinstance(dict2[list(dict2.keys())[0]], list):
        dict2_len = len(dict2[list(dict2.keys())[0]])

    # Merge two dictionaries
    for key in np.unique(list(dict1.keys()) + list(dict2.keys())):
        key = str(key)

        if key == 'galaxy_catalogues':
            dict2[key] = np.array(dict2[key], dtype=object)

        # If the secondary dict has a key not in the primary, pad with Nones
        if key not in dict1 and np.ndim(dict2[key]) > 0 and isinstance(dict2[key][0], ndarray):
            dict1[key] = [np.array([None] * len(dict2[key][0]))] * dict1_len
        elif key not in dict1:
            dict1[key] = [None] * dict1_len

        # If the primary dict has a key not in the secondary dict, pad with Nones, else merge dicts
        if key not in dict2 and isinstance(dict1[key][0], ndarray):
            dict1[key].extend([np.array([None] * len(dict1[key][0]))] * dict2_len)
        elif key not in dict2:
            dict1[key].extend([None] * dict2_len)
        elif np.ndim(dict2[key]) > 0:
            dict1[key].extend(dict2[key])
        else:
            dict1[key].append(dict2[key])
    return dict1


def list_dict_convert(
        data: list[dict[str, float | ndarray]],
) -> dict[str, list[float | None] | list[ndarray]]:
    """
    Converts a list of dictionaries to a dictionary of lists

    Parameters
    ----------
    data : list[dict[str, float | ndarray]]
        List of dictionaries to convert

    Returns
    -------
    dict[str, list[float | None] | list[ndarray]]
        Dictionary of lists
    """
    value: dict[str, float | ndarray]
    new_data: dict[str, list[float] | list[ndarray]] = {}

    for value in data:
        dict_list_append(new_data, value)

    return new_data


def main(dir_path: str, overwrite: bool = False, save_path: str = '') -> None:
    """
    Takes all pickle files in a directory and subdirectories of labels and images and combines them
    into a single pickle file

    Parameters
    ----------
    dir_path : str
        Path to the directory
    overwrite : bool, default = False
        If a file with the same save name already exists, should it be overwritten
    save_path : str, default = ''
        Path to save the combined pickle file, if empty, will not save
    """
    root: str
    files: list[str]
    value: list[float] | list[ndarray]
    image: list[ndarray] | ndarray
    images: list[ndarray] | ndarray = []
    label: (dict[str, float | list[float] | list[ndarray] | ndarray] |
            list[dict[str, float | ndarray]])
    labels: dict[str, list[float | None] | list[ndarray] | ndarray] = {}
    file: BinaryIO

    # Loop through all files in the directory and subdirectories
    for root, _, files in os.walk(dir_path):
        for name in files:
            label, image = load_pickle(f'{root}/{name}')

            # If the label is a list of dictionaries, convert to dictionary of lists
            if isinstance(label, list):
                label = list_dict_convert(label)

            # Merge props dictionary into label dictionary
            if 'props' in label and isinstance(label['props'], list):
                label = label | list_dict_convert(label['props'])
                del label['props']
            elif 'props' in label:
                label = label | label['props']
                del label['props']

            # Merge label into all labels
            dict_list_append(labels, label)

            # Merge image into all images
            if np.ndim(image) == 4:
                images.extend(image)
            else:
                images.append(image)

    # Convert lists to numpy arrays
    for key, value in labels.items():
        labels[key] = np.array(value, dtype=object if key == 'galaxy_catalogues' else None)

    # Make sure shape is (N,C,H,W)
    images = np.array(images)

    if images.shape[-1] != images.shape[-2]:
        images = np.moveaxis(images, -1, 1)

    # Normalise images and save normalisations in labels
    labels['norms'] = np.max(images, axis=(-2, -1))
    images /= labels['norms'][..., np.newaxis, np.newaxis]

    if not save_path:
        return

    if not overwrite and os.path.exists(save_path):
        getLogger(__name__).error(f'{save_path} already exists and overwrite is False, file will '
                                  f'not be saved')
        return

    with open(save_path, 'wb') as file:
        pickle.dump((labels, images), file)


if __name__ == "__main__":
    main(
        '../data/temp/',
        overwrite=True,
        save_path='../data/darkskies_0.07.pkl',
    )
