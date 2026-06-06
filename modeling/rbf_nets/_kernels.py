""" Kernels for the RBF Networks

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2024
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric',
    'inverse_quadratic', 'inverse_multiquadric', 'gaussian', 'get_ker_func']

import numpy as np


##############################################################################
##                          Radial Basis Functions                          ##
##############################################################################

def linear(r):
    """ Linear RBF (y(r) = -r) """
    return -r

def thin_plate_spline(r):
    """ Thin plate spline RBF (y(r) = r² * log(r)) """
    return r**2 * np.log(r)

def cubic(r):
    """ Cubic RBF (y(r) = -r³) """
    return r**3

def quintic(r):
    """ Quintic RBF (y(r) = -r⁵) """
    return -r**5

def multiquadric(r):
    """ Multiquadratic RBF (y(r) = -√(1+r²) """
    return -np.sqrt(1. + r**2)

def inverse_quadratic(r):
    """ Inverse Quadratic RBF (y(r) = 1/(1+r²)) """
    return 1./(1. + r**2)

def inverse_multiquadric(r):
    """ Inverse Multiquadratic RBF (y(r) = 1/√(1+r²) """
    return 1./np.sqrt(1. + r**2)

def gaussian(r):
    """ Gaussian RBF (y(r) = -exp(-r²)) """
    return np.exp(-r**2)

def get_ker_func(kernel):
    """ Get the kernel function from its name

    Take the name of a kernel and return a reference to the corresponding
    function; raise a `ValueError` if the name is unknown.

    Parameters
    ----------
    kernel : str
        The type of RBFs to be used. Possibilities:
          - 'linear': y(r) = -r
          - 'thin_plate_spline': y(r) = r² * log(r)
          - 'cubic': y(r) = -r³
          - 'quintic': y(r) = -r⁵
          - 'multiquadric': y(r) = -√(1+r²)
          - 'inverse_multiquadric': y(r) = 1/√(1+r²)
          - 'inverse_quadratic': y(r) = 1/(1+r²)
          - 'gaussian': y(r) = -exp(-r²)
            :Default: 'linear'

    Returns
    -------
    fker : reference to a function
        The kernel function.

    Examples
    --------
    # Linear kernel
    >>> kernel = get_kernel('linear')
    >>> kernel(1.23)
    -1.23

    # Gaussian kernel
    >>> kernel = get_kernel('gaussian')
    >>> kernel(1.23)
    0.22027026705244174
    """
    # pylint: disable=too-many-return-statements

    # Get the kernel function
    ker = kernel.lower()
    if ker == 'linear':
        return linear
    if ker == 'thin_plate_spline':
        return thin_plate_spline
    if ker == 'cubic':
        return cubic
    if ker == 'quintic':
        return quintic
    if ker == 'multiquadric':
        return multiquadric
    if ker == 'inverse_multiquadric':
        return inverse_multiquadric
    if ker == 'inverse_quadratic':
        return inverse_quadratic
    if ker == 'gaussian':
        return gaussian

    # Raise an error if the kernel is invalid
    raise ValueError(f"Wrong value `{kernel}` for `kernel`; options are:\n"
        + "\t{'linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric',\n"
        + "\t 'inverse_multiquadric', 'inverse_quadratic', 'gaussian'}")

##############################################################################
