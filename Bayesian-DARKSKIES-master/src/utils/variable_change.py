"""
Converts between cross-section and model
"""
from copy import deepcopy as cp

import numpy as np
from numpy import ndarray
from astropy.units.core import Unit
from astropy.modeling.physical_models import NFW


def sigma0_equation(velocity: ndarray, sigma: float | ndarray, w: float = 560) -> ndarray:
    """
    Calculates sigma0 from the cross-section

    Parameters
    ----------
    velocity : ndarray
        Virial velocities of the clusters
    sigma : float | ndarray
        sigma of the clusters
    w : float, default = 560
        Turnover velocity (from the Robertson model)

    Returns
    -------
    ndarray
        sigma0 for the clusters
    """
    return sigma * (1 + (velocity / w) ** 2)


def sigma_vd(velocity: ndarray, sigma0: float | ndarray = 3.04, w: float = 560) -> ndarray:
    """
    Calculates cross-section from sigma0

    Parameters
    ----------
    velocity : ndarray
        Virial velocities of the clusters
    sigma0 : float | ndarray, default = 3.04
        sigma0 of the clusters
    w : float, default = 560
        Turnover velocity (from the Robertson model)

    Returns
    -------
    ndarray
        Cross-section for the clusters
    """
    return sigma0 / (1 + velocity ** 2 / w ** 2)


def convert_param_space(
        mass: float | ndarray,
        cross_section: float | ndarray,
        w: float = 560,
        cdm: float = 0.05) -> ndarray:
    """
    Convert a cluster simulated in a velocity independent
    space to a velocity dependent space.

    assuming that

    sigma = sigma_0 ( 1 + v**2 / w**2 )**-1

    and that we are working with clusters, so we can assume
    that the velocity > w such that we can fix w since we
    are not sensitive to this any way.

    We fix w=560km/s

    We use the virial theorem to convert between mass
    and velocity

    inputs :
        mass : either a single float or numpy array of floats
               of the absolute virial mass
        cross_section : float or numpy array (matching mass)
               of the cross-section jof the cluster
    keywords:
        w : the assumed turnover velocity (assumed to be that of Robertson model)
        cdm : the value for an input cross-section of 0 which is unphysical
    returns
        sigma_0 : the new parameter, the normalisation of the
                velocity dependent cross-section

    """
    use_cross_section = cp(cross_section)  # because if I change it, it changes the pointed to array

    if not isinstance(mass, (ndarray, float)):
        raise ValueError(f'{type(mass)} for mass not supported')

    if not isinstance(use_cross_section, (ndarray, float)):
        raise ValueError(f'{type(use_cross_section)} for cross_section not supported')

    if isinstance(use_cross_section, ndarray):
        use_cross_section[use_cross_section == 0] = cdm
    elif use_cross_section == 0:
        use_cross_section = cdm

    if np.any(mass < 1e10):
        raise ValueError("Some (or all) masses are too low make sure you are giving absolute mass")

    # Create a bunch of NFW halos since this contains nice functions that we can use
    if isinstance(mass, ndarray):
        nfw_objects = [NFW(mass=i * Unit('solMass'), massfactor='virial') for i in mass]
    else:
        nfw_objects = [NFW(mass=mass * Unit('solMass'), massfactor='virial')]

    # Use its inbuilt function to find velocities
    virial_velocities = np.array([nfw_object.circular_velocity(
        nfw_object.r_virial.to(Unit('kpc')),
    ).to_value() for nfw_object in nfw_objects])

    # return the function converting
    return sigma0_equation(virial_velocities, use_cross_section, w=w)


def convert_param_space_vd(
        mass: float | ndarray,
        w: float = 560) -> ndarray:
    """
    Convert a cluster simulated in a velocity dependent
    space to a velocity independent space.

    assuming that

    sigma = sigma_0 ( 1 + v**2 / w**2 )**-1

    and that we are working with clusters, so we can assume
    that the velocity > w such that we can fix w since we
    are not sensitive to this any way.

    We fix w=560km/s

    We use the virial theorem to convert between mass
    and velocity

    inputs :
        mass : either a single float or numpy array of floats
               of the absolute virial mass
    keywords:
        w : the assumed turnover velocity (assumed to be that of Robertson model)
    returns
        cross-section

    """
    if not isinstance(mass, (ndarray, float)):
        raise ValueError(f'{type(mass)} for mass not supported')

    if np.any(mass < 1e10):
        raise ValueError("Some (or all) masses are too low make sure you are giving absolute mass")

    # Create a bunch of NFW halos since this contains nice functions that we can use
    if isinstance(mass, ndarray):
        nfw_objects = [NFW(mass=i * Unit('solMass'), massfactor='virial') for i in mass]
    else:
        nfw_objects = [NFW(mass=mass * Unit('solMass'), massfactor='virial')]

    # Use its inbuilt function to find velocities
    virial_velocities = np.array([nfw_object.circular_velocity(
        nfw_object.r_virial.to(Unit('kpc')),
    ).to_value() for nfw_object in nfw_objects])

    # return the function converting
    return sigma_vd(virial_velocities, w=w)
