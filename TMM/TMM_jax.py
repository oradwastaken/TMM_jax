#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMM_jax.py

Imports functions from the main TMM file and produces jax.numpy versions that can be jitted or auto-differentiated.
"""

import jax

from TMM import TMM_main


def TMM_jax(
    wavelengths,
    thicknesses,
    indices,
    angles=0,
    n_BG_in=1.0,
    n_BG_out=None,
    polarization="p",
):
    TMM_main.np = jax.numpy
    return TMM_main.TMM(
        wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
    )


def calc_M_matrix_jax(
    wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
):
    TMM_main.np = jax.numpy
    return TMM_main.calc_M_matrix(
        wavelengths, thicknesses, indices, angles, n_BG_in, n_BG_out, polarization
    )


TMM_jit = jax.jit(TMM_jax, static_argnums=6)
calc_M_matrix_jit = jax.jit(calc_M_matrix_jax, static_argnums=6)
