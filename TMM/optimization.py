#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMM/optimization.py

Contains some helper functions to perform TMM optimizations.
"""

from copy import deepcopy
from itertools import product


from . import generate_index_bilayer


def get_default_TMM_params(args):
    """Takes an args object (imported from config.py), and returns a dict with the default TMM parameters described
    there. The only parameter that isn't included is thicknesses, since we are expecting to optimize over those
    values.
    """
    indices = generate_index_bilayer(
        args.n1, args.n2, args.wavelengths, args.num_layers
    )

    TMM_default_params = {
        "wavelengths": args.wavelengths,
        "angles": args.angles,
        "indices": indices,
        "n_BG_in": args.n_BG_in,
        "n_BG_out": args.n_BG_out,
        "polarization": args.polarization,
    }

    return TMM_default_params


def generate_args_list(args, sweep_targets):
    """Takes an args object (imported from config.py) and a dict of values to sweep. For example:
        sweep_targets = {'num_layers': [5, 10, 15]}

    This function returns a list of args objects, where each object has the attributes listed as dictionary keys (so,
    num_layers in this example) are replaced by the listed values. In this example, we would end up with a list of 3
    sets of args, with the default value of num_layers replaced by num_layers = 5, num_layers = 10, and num_layers = 15.

    This works for any number of key-value pairs in the dict, and it even multiplexes them to perform multidimensional
    sweeps.
    """
    if sweep_targets is None:
        return [args]

    attrs_to_sweep = list(sweep_targets.keys())
    sweep_values = list(sweep_targets.values())
    combinations = list(product(*sweep_values))
    print(f"Total combinations: {len(combinations)}")

    for new_values in combinations:
        temp_args = deepcopy(args)
        for value, attr in zip(new_values, attrs_to_sweep):
            setattr(temp_args, attr, value)
            yield temp_args
