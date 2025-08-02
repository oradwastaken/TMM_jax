#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_global_optimizer_example.py

An example optimizer for a transfer-matrix method (TMM) calculator, enhanced using Google JAX
"""

from functools import partial
import random

import jax.numpy as np
import jax
from jax import jit, grad
import matplotlib.pyplot as plt
from scipy import optimize

from TMM import TMM_jit
from materials import n_Si, n_SiO2


def main():
    # We aren't planning on using the GPU yet. This prevents JAX from looking for it.
    jax.config.update("jax_platform_name", "cpu")
    # Supposedly, if you have CUDA, you can easily distribute this to the GPU

    # If run on a 64-bit machine, tell jax explicitly:
    jax.config.update("jax_enable_x64", True)

    # Set up the initial simulation:
    num_layers = 21

    wavelengths_target = np.linspace(1.55, 1.56, 21)
    indices = [
        n_Si(wavelengths_target)
        if is_even(layer_number)
        else n_SiO2(wavelengths_target)
        for layer_number in range(num_layers)
    ]
    TMM_params = {
        "wavelengths": wavelengths_target,
        "indices": indices,
        "n_BG_in": 1.0,
        "n_BG_out": 1.0,
        "angles": 0,
        "polarization": "p",
    }

    # Define optimization functions:
    target_R = np.ones_like(wavelengths_target)
    TMM_thickness_only = partial(TMM_jit, **TMM_params)
    num_points = len(wavelengths_target)

    def obj_func(thicknesses):
        results = TMM_thickness_only(thicknesses=thicknesses)
        loss = np.sqrt(np.sum(np.abs(results.R - target_R) ** 2 / num_points))
        return loss

    grad_func = grad(obj_func)
    obj_func_compiled = jit(obj_func)
    grad_func_compiled = jit(grad_func)

    min_index = np.real(np.array(indices)).min()
    max_wavelength = wavelengths_target.max()
    bounds = (0, 4 * max_wavelength / min_index)
    thickness_guess = np.array([random.uniform(*bounds) for _ in range(num_layers)])

    # # Local optimization:
    # Uncomment for a local optimizer
    # optimization_result = optimize.minimize(fun=obj_func_compiled, x0=thickness_guess, jac=grad_func_compiled,
    #                                         bounds=[bounds] * num_layers, method='TNC')

    # Global? optimization:
    optimization_result = optimize.basinhopping(
        func=obj_func_compiled,
        x0=thickness_guess,
        niter=500,
        minimizer_kwargs={
            "jac": grad_func_compiled,
            "bounds": [bounds] * num_layers,
            "method": "TNC",
        },
    )

    print(f"Optimized layer thicknesses: {optimization_result.x}")

    # Now let's plot the result for a broad bandwidth:
    wavelength_broadband = np.linspace(1.5, 1.6, 401)
    indices_broadband = [
        n_SiO2(wavelength_broadband) if i % 2 == 0 else n_Si(wavelength_broadband)
        for i in range(num_layers)
    ]
    TMM_params["wavelengths"] = wavelength_broadband
    TMM_params["indices"] = indices_broadband
    TMM_params["thicknesses"] = optimization_result.x
    optimized_TMM_result = TMM_jit(**TMM_params)

    plt.plot(
        optimized_TMM_result.wavelengths, optimized_TMM_result.R, label="Best device"
    )
    plt.plot(wavelengths_target, target_R, "x", label="target")
    plt.xlabel("wavelength (µm)")
    plt.xlabel("reflectivity")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.show()


def is_even(number) -> bool:
    """Helper function to determine if a number is even"""
    return True if number % 2 == 0 else False


if __name__ == "__main__":
    main()
