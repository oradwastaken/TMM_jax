#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_example_TMM.py

An example demonstrating how to run a simple TMM calculation.
"""

import matplotlib.pyplot as plt
import numpy as np

from TMM import TMM, generate_index_bilayer
from materials import n_Si, n_SiO2


def main():
    wavelengths = np.linspace(1.5, 2, 1001)
    thicknesses = np.array([1, 2, 1])
    indices = generate_index_bilayer(n_Si, n_SiO2, wavelengths, len(thicknesses))

    result = TMM(wavelengths=wavelengths, thicknesses=thicknesses, indices=indices)

    plt.figure()
    plt.plot(result.wavelengths, result.T)
    plt.xlabel("wavelength (µm)")
    plt.ylabel("transmittance")
    plt.xlim(wavelengths.min(), wavelengths.max())
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
