#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_materials.py

Contains functions that return optical constants for many common optical materials.
"""

import numpy as np


def main():
    """If you run this file, it will plot one of the materials :)"""
    import matplotlib.pyplot as plt

    wavelength = np.linspace(3, 5, 101)
    index = n_SiO2(wavelength)
    plot_title = "n_SiO2"

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(wavelength, index.real)
    plt.xlabel("wavelength (um)")
    plt.ylabel("refractive index (real)")

    plt.subplot(1, 2, 2)
    plt.plot(wavelength, index.imag)
    plt.xlabel("wavelength (um)")
    plt.ylabel("refractive index (imag)")
    plt.ylim(bottom=0)

    plt.suptitle(plot_title)

    plt.tight_layout()
    plt.show()


def sanitize_wavelength(wavelength, unit, min_wavelength, max_wavelength):
    """takes wavelength and converts to um, and to a numpy array"""

    if not isinstance(wavelength, np.ndarray):
        wavelength = np.asanyarray(wavelength)

    if unit == "nm":
        wavelength = wavelength * 10**-3
    elif unit == "m":
        wavelength = wavelength * 10**6
    elif unit == "um":
        pass
    else:
        raise Exception("The unit doesn't make sense. Accepts nm, um, or m.")

    if wavelength.min() < min_wavelength:
        raise Exception(
            f"Shortest wavelength below acceptable range for this material (lambda = {min_wavelength} um)."
        )
    elif wavelength.max() > max_wavelength:
        raise Exception(
            f"Longest wavelength above acceptable range for this material (lambda = {max_wavelength} um)."
        )

    return wavelength


def n_from_CSV(wavelength, filename):
    """Imports the refractive index from a csv and returns a complex numpy array. Expects wavelengths in um."""

    my_data = np.genfromtxt(filename, delimiter=",", skip_header=1)

    wavelength_exp = my_data[:, 0]
    n_real_exp = my_data[:, 1]
    n_imag_exp = my_data[:, 2]

    n_real_out = np.interp(wavelength, wavelength_exp, n_real_exp)
    n_imag_out = np.interp(wavelength, wavelength_exp, n_imag_exp)

    return n_real_out - 1j * n_imag_out


def n_Al(wavelength, *, unit="um"):
    """
    Optical constants for Al
    Source: Rakic 1995, https://doi.org/10.1364/AO.34.004755

    Example:
        >>> n_Al(1.0)
        (1.44-9.49j)
    """

    min_acceptable_wavelength = 0.000124
    max_acceptable_wavelength = 200.0

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    return n_from_CSV(wavelength=wavelength, filename="materials/Al_Rakic.csv")


def n_Au(wavelength, *, unit="um"):
    """
    Optical constants for Au
    Source: P. B. Johnson and R. W. Christy 1972, https://doi.org/10.1103/PhysRevB.6.4370

    Example:
        >>> n_Au(1.0)
        (0.2276923076923077-6.473076923076923j)
    """

    min_acceptable_wavelength = 0.1879
    max_acceptable_wavelength = 1.937

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    return n_from_CSV(wavelength=wavelength, filename="materials/Au_Johnson.csv")


def n_Ge(wavelength, *, unit="um"):
    """
    Optical constants for Ge
    Source: Amotchkina 2020, https://doi.org/10.1364/AO.59.000A40
    Data lifted from here: https://refractiveindex.info/?shelf=main&book=Ge&page=Amotchkina

    Example:
        >>> n_Ge(4.5)
        (3.97621+0j)
    """

    min_acceptable_wavelength = 0.4
    max_acceptable_wavelength = 11.0

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    return n_from_CSV(wavelength=wavelength, filename="materials/Ge_Amotchkina.csv")


def n_InGaAs(wavelength, *, unit="um"):
    """
    Optical constants for In(1-x)Ga(x)As; x=0.48
    Source: Adachi 1989, https://doi.org/10.1063/1.343580

    Code lifted from here: https://github.com/polyanskiy/refractiveindex.info-scripts/blob/master/scripts/Adachi%201989%20-%20InGaAs.py

    Example:
        >>> n_InGaAs(1.0)
        (3.52+0.13617391304347828j)
    """

    min_acceptable_wavelength = 0.207
    max_acceptable_wavelength = 12.4

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    return n_from_CSV(wavelength=wavelength, filename="materials/InGaAs_Adachi.csv")


def n_InP(wavelength, *, unit="um"):
    """
    Optical constants for InP
    Source: Adachi 1989, https://doi.org/10.1063/1.343580

    Code lifted from here: https://github.com/polyanskiy/refractiveindex.info-scripts/blob/master/scripts/Adachi%201989%20-%20InP.py

    Example:
        >>> n_InP(0.5876,'um')
        (3.35+0.3034j)
    """

    min_acceptable_wavelength = 0.207
    max_acceptable_wavelength = 12.4

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    return n_from_CSV(wavelength=wavelength, filename="materials/InP_Adachi.csv")


def n_PMMA(wavelength, *, unit="um"):
    """
    Optical constants of (C5H8O2)n (Poly(methyl methacrylate), PMMA)
    Source: X. Zhang 2020, https://doi.org/10.1364/AO.383831
            X. Zhang 2020, https://doi.org/10.1016/j.jqsrt.2020.107063

    Example:
        >>> n_PMMA(0.5)
        (1.49021+2.24e-07j)
    """

    min_acceptable_wavelength = 0.4
    max_acceptable_wavelength = 19.94

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    return n_from_CSV(wavelength=wavelength, filename="materials/PMMA_Zhang.csv")


def n_Si(wavelength, *, unit="um"):
    """
    Optical constants for Si

    For wavelengths between 1.357 and 11 µm:
    Ref: B. Tatian. Fitting refractive-index data with the Sellmeier dispersion formula, Appl. Opt. 23, 4477-4485 (1984)
    Source: https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg

    For wavelengths between 0.25 and 1.357 µm:
    Ref: C. Schinke. Uncertainty analysis for the coefficient of band-to-band absorption of crystalline silicon. AIP Advances 5, 67168 (2015)
    Source: https://refractiveindex.info/?shelf=main&book=Si&page=Schinke

    Example:
        >>> n_Si(1.55)
        3.477723756220899
    """

    min_acceptable_wavelength = 0.25
    max_acceptable_wavelength = 11.04

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    # Separate into long and short wavelength ranges:
    wavelength_long = wavelength[wavelength >= 1.357]
    wavelength_short = wavelength[wavelength < 1.357]

    C1 = 10.6684293
    C2 = 0.301516485
    C3 = 0.0030434748
    C4 = 1.13475115
    C5 = 1.54133408
    C6 = 1104

    n_out_long = np.sqrt(
        1
        + C1 * wavelength_long**2 / (wavelength_long**2 - C2**2)
        + C3 * wavelength_long**2 / (wavelength_long**2 - C4**2)
        + C5 * wavelength_long**2 / (wavelength_long**2 - C6**2)
    )

    n_out_short = n_from_CSV(
        wavelength=wavelength_short, filename="materials/Si_Schinke.csv"
    )

    n_out = np.append(n_out_short, n_out_long)
    n_out = n_out if len(n_out) > 1 else n_out[0]

    return n_out


def n_SiO2(wavelength, *, unit="um"):
    """
    Optical constants for SiO2 based on Sellmeier coefficients
    Source: http://refractiveindex.info/?group=CRYSTALS&material=SiO2

    Example:
        >>> n_SiO2(633, unit='nm')
        1.44402362
    """

    min_acceptable_wavelength = 0.21
    max_acceptable_wavelength = 6.7

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    C1 = 0.6961663
    C2 = 0.0684043
    C3 = 0.4079426
    C4 = 0.1162414
    C5 = 0.8974794
    C6 = 9.896161

    n_out = np.sqrt(
        1
        + C1 * wavelength**2 / (wavelength**2 - C2**2)
        + C3 * wavelength**2 / (wavelength**2 - C4**2)
        + C5 * wavelength**2 / (wavelength**2 - C6**2)
    )

    return n_out


def n_TiO2(wavelength, *, unit="um"):
    """
    Optical constants for TiO2
    Source: Sarkar 2019, https://refractiveindex.info/?shelf=main&book=TiO2&page=Sarkar
    S. Sarkar, Hybridized guided-mode resonances via colloidal plasmonic self-assembled grating, ACS Appl. Mater. Interfaces, 11, 13752-13760 (2019)

    Example:
        >>> n_TiO2(0.35,'um')
        (2.585271+0.029085j)
    """

    min_acceptable_wavelength = 0.3
    max_acceptable_wavelength = 1.69

    wavelength = sanitize_wavelength(
        wavelength, unit, min_acceptable_wavelength, max_acceptable_wavelength
    )

    return n_from_CSV(wavelength=wavelength, filename="materials/TiO2_Sarkar.csv")


if __name__ == "__main__":
    main()
