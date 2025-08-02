#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMM_jax package

Includes classes and subroutines to calculate transmittance and reflectance of a multilayer thin-film stack
using the Transfer Matrix Method (TMM). Based on the approach described in Ch 10 of Steck:

Steck, D. A. Classical and Modern Optics. (2006)
(available online at http://steck.us/teaching)
"""

from .TMM_main import (
    TMM as TMM,
    generate_index_bilayer as generate_index_bilayer,
    calc_M_matrix as calc_M_matrix,
)
from .TMM_jax import (
    TMM_jit as TMM_jit,
    TMM_jax as TMM_jax,
    calc_M_matrix_jax as calc_M_matrix_jax,
    calc_M_matrix_jit as calc_M_matrix_jit,
)
