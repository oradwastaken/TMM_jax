#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMM_main.py

Speed test results:
75 layers, 2 materials, no dispersion
21 angles, 1 wavelength, 1 polarizaiton

Old numba @njit code:
1000 loops, avg: 1.31 ms; best time: 0.98 ms

Equivalent in MATLAB:
1000 loops, avg: 33.9 ms

Equivalent in Lumerical:
1000 loops, avg: 5.59 ms

numpy (old code)
1000 loops, avg: 25.01 ms

numpy (this iteration):
1000 loops, avg: 0.44 ms

JAX: (unjitted)
1000 loops, 20.1 ms

JAX: (jitted)
1000 loops, 1.26 ms
"""

from collections import namedtuple
from functools import reduce
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .Polarization import Polarization

# Define types:
float_array = ArrayLike
complex_array = ArrayLike

TMMArguments = namedtuple(
    "TMMArguments",
    "indices thicknesses wavelengths angles_rad n_BG_in n_BG_out polarization",
)

TMMResults = namedtuple("TMMResults", "wavelengths angles r t R T A")


def TMM(
    wavelengths: float_array,
    thicknesses: float_array,
    indices: complex_array,
    angles: float_array = 0,
    n_BG_in: float = 1,
    n_BG_out: float = None,
    polarization: str = "p",
):
    """Performs the TMM calculation, and calculates the reflection and transmission.

    Returns a TMMResult named tuple with the following attributes, in order: wavelengths, angles, r, t, R, T, A.
    The last 5 results are each arrays of size (num_wavelengths, num_angles)

    This pure python/Jax TMM_uncompiled() function is fairly slow. For performance, either import TMM(), or rejit the
    function in your main() file using TMM = jax.jit(TMM_uncompiled, static_argnums=6)
    """
    M = calc_M_matrix(
        wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
    )

    # First, reshape arrays to have the necessary dimensions for broadcasting:
    args = _convert_to_np(
        wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
    )
    wavelengths = np.atleast_2d(args.wavelengths)
    incoming_angle_rad = np.atleast_2d(args.angles_rad).transpose()
    outgoing_angle_rad = _calc_angle_in_medium(
        args.n_BG_out, args.angles_rad, args.n_BG_in
    )

    alpha_in = _calc_alpha(args.n_BG_in, incoming_angle_rad, args.polarization)
    alpha_out = _calc_alpha(args.n_BG_out, outgoing_angle_rad, args.polarization)
    k_in = (
        2 * np.pi * args.n_BG_in * np.cos(incoming_angle_rad) / wavelengths
    ).transpose()
    k_out = (
        2 * np.pi * args.n_BG_out * np.cos(outgoing_angle_rad) / wavelengths
    ).transpose()

    M00 = M[0, 0, ...]
    M01 = M[0, 1, ...]
    M10 = M[1, 0, ...]
    M11 = M[1, 1, ...]

    t = (
        2
        * alpha_in
        / (alpha_in * M00 + alpha_in * alpha_out * M01 + M10 + alpha_out * M11)
    )
    r = (alpha_in * M00 + alpha_in * alpha_out * M01 - M10 - alpha_out * M11) / (
        alpha_in * M00 + alpha_in * alpha_out * M01 + M10 + alpha_out * M11
    )
    T = np.abs(t) ** 2 * np.real(k_out / k_in)
    R = np.abs(r) ** 2
    A = 1 - R - T

    # Explicitly using numpy instead of jax.numpy so that these values are converted back into numpy arrays
    return TMMResults(
        np.squeeze(wavelengths),
        np.squeeze(np.rad2deg(incoming_angle_rad)),
        np.squeeze(r),
        np.squeeze(t),
        np.squeeze(R),
        np.squeeze(T),
        np.squeeze(A),
    )


def _convert_to_np(
    wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
):
    """Takes the input parameters and massages them into the correct jax.numpy array types and shapes,
    and returns a TMMArguments named tuple object.
    """
    # By wrapping everything with "jax.numpy.array," we force-convert all inputs to jax.numpy:
    wavelengths = np.array(np.atleast_1d(wavelengths))  # in um
    thicknesses = np.array(np.atleast_1d(thicknesses))  # in um
    indices = np.array(np.conj(np.atleast_1d(indices)))  # index of each layer
    n_BG_in = np.array(
        np.atleast_1d(n_BG_in)
    )  # index of background, defaults to air (n=1)
    angles_rad: float_array = np.array(
        np.deg2rad(np.atleast_1d(angles))
    )  # list of incident angles, in rad

    # By default, if unspecified, n_BG_out should be the same as n_BG_in
    if n_BG_out is None:
        n_BG_out = n_BG_in
    else:
        n_BG_out = np.array(np.atleast_1d(n_BG_out))

    # Convert to a Polarization Enum type if not one already
    if not isinstance(polarization, Polarization):
        polarization = Polarization.from_string(polarization)

    return TMMArguments(
        indices, thicknesses, wavelengths, angles_rad, n_BG_in, n_BG_out, polarization
    )


def _calc_angle_in_medium(n: complex_array, angles_rad, n_BG_in) -> float_array:
    """Takes the incident angle (from a medium of index n_BG_in) and converts it into the angle
    inside that of a medium of refractive index n.
    Returns an array of shape: (num_wavelengths, num_angles)
    """
    n = np.atleast_2d(n)
    angles_rad = np.atleast_2d(angles_rad).transpose()
    theta_in_medium = np.array(np.arcsin(n_BG_in / n * np.sin(angles_rad)))

    # Test for stability, and fix unstable points to the correct branch (theta -> pi - theta)
    # For more info, see Appendix D of arXiv:1603.02720 [physics.comp-ph] https://arxiv.org/abs/1603.02720
    unstable_angles = np.imag(np.conj(n) * np.cos(theta_in_medium)) < 0
    theta_in_medium = np.where(
        unstable_angles, np.pi - theta_in_medium, theta_in_medium
    )
    return theta_in_medium


def _calc_alpha(
    n: complex, angle_in_medium: complex_array, polarization: Polarization
) -> complex_array:
    """Calcualtes the 'geometry coefficient' alpha for a given single layer. Here, angle_in_medium is the propagation
    angle in radians INSIDE the medium, and may be complex if the incident angle is too steep.
    Returns an array of shape: (num_wavelengths, num_angles)
    """
    match polarization:
        case Polarization.p:
            return (n / np.cos(angle_in_medium)).transpose()
        case Polarization.s:
            return (n * np.cos(angle_in_medium)).transpose()


def _calc_F_matrix(
    n, L, wavelengths, angles_rad, n_BG_in, polarization
) -> complex_array:
    """Given the characteristic impedance alpha and the phase phi of a given layer, calculates the characteristic
    matrix (2, 2, num_wavelengths, num_angles) for a single given layer.
    """
    angle_in_medium = _calc_angle_in_medium(n, angles_rad, n_BG_in)
    alpha = _calc_alpha(n, angle_in_medium, polarization)
    phi = (2 * np.pi * (n / wavelengths) * L * np.cos(angle_in_medium)).transpose()

    F11 = np.cos(phi)
    F22 = F11
    F12 = -1j * np.sin(phi) / alpha
    F21 = F12 * alpha**2
    F = np.array([[F11, F12], [F21, F22]])
    return F


def calc_M_matrix(
    wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
):
    """Calculates the individual characteristic matrices F for all layers, then multiplies them all up
    to return the M matrix for the complete stack. M is an array of shape (2, 2, num_wavelengths, num_angles).
    """
    args = _convert_to_np(
        wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
    )

    F_list = [
        _calc_F_matrix(
            n, L, args.wavelengths, args.angles_rad, args.n_BG_in, args.polarization
        )
        for (n, L) in zip(args.indices, args.thicknesses)
    ]

    # %% Multiply up all the Fs, i.e., M = F1 * F2 * F3...
    M = reduce(mat_product, F_list)
    return M


def generate_index_bilayer(
    n1: Callable, n2: Callable, wavelengths: float_array, num_layers: int
):
    """Creates an index array that can be the indices argument for the TMM function. It alternates 2 refractive
    indices for num_layers number of layers.

    Here, n1 and n2 are *functions* that return the refractive index of materials. For example,
    n_SiO2(0.633) = 1.44402362. If wavelengths is an array, n1 should also return an array. This function returns
    a 2D list, so something o the form:

                    layer 1                |                layer 2                | ....
    [[n1(wavelength1), n1(wavelength2)...],  [n2(wavelength1), n2(wavelength2)...], ....]
    """
    indices = [
        n1(wavelengths) if is_even(layer_number) else n2(wavelengths)
        for layer_number in range(num_layers)
    ]
    return indices


def is_even(number):
    return True if number % 2 == 0 else False


def mat_product(a, b):
    """Performs matrix multiplication using einsum, e.g., a @ b"""
    return np.einsum("ijml, jkml -> ikml", a, b)
