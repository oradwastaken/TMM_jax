# TMM_jax

TMM_jax is a global optimizer for the transfer matrix method (TMM). It uses both [autograd](https://en.wikipedia.org/wiki/Automatic_differentiation) 
and [just-in-time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation)
(jit) from the [Google JAX](https://github.com/google/jax) library. Combined, 
these two things make optimizing TMM blazing fast. JAX also supports 
vectorization, and parallelization using GPU support. (Neither feature is
implemented in TMM_jax.)

The TMM formalism is based on the approach described by Steck:

* Steck, Daniel A. "Ch 10: Thin Films." _Classical and Modern Optics_, 2006, p. 174 - 178. Available online at http://steck.us/teaching.


## Installation

Package maintenance is done using [uv](https://docs.astral.sh/uv/). To install the package, clone the repo and run `uv sync`. To run, `uv run python main_example_TMM.py`.

## Sharp edges
As a consequence of writing the code using jax.numpy instead of pure numpy, 
all arrays need to be reinterpreted for JAX. Often, this means you need to create
arrays and stuff using the JAX version of numpy directly:

    import jax.numpy as np 
    a = np.array(...)

instead of 

    import numpy as np 
    a = np.array(...)

You want to avoid doing this as much as possible, since jax.numpy has a 
lot of overhead, [making many routine array operations needlessly 
slow](https://jax.readthedocs.io/en/test-docs/faq.html#creating-arrays-with-jax-numpy-array-is-slower-than-with-numpy-array).
If you have an odd error you don't know how to debug, try making this 
substitution.

One way to speed up code is to use pure numpy to _create_ things, and wrap them in
jax arrays just before compilation. For example:

    import numpy as np
    import jax.numpy as jnp
    
    a = np.array([[1, 2, 3], [4, 5, 6], ...])  # And then finally:
    a = jnp.array(a)

## TMM
To perform a TMM calculation, import one of the `TMM` functions from the TMM package:

    from TMM import TMM       # pure numpy version
    from TMM import TMM_jax   # un-jitted JAX version (slow)
    from TMM import TMM_jit   # jitted JAX version (fast, but with setup overhead)

An example of how to use the `TMM()` functions can be found in main_example_TMM.py.

The arguments to the functions are as follows:

    TMM(wavelengths, thicknesses, indices, angles, 
           n_BG_in, n_BG_out, polarization,)

where the types are as follows:
* `wavelengths`: `list[float]` — operating wavelength in µm.
* `thicknesses`: `list[float]` — layer thicknesses in µm.
* `indices`: `list[list[complex]]` — refractive index vs. wavelength, one for each layer.
* `angles`: `list[float]` — incident angle degrees. Defaults to normal incidence.
* `n_BG_in`: `float` — refractive index of incoming beam side. Defaults to n = 1.
* `n_BG_out`: `float` — refractive index of outgoing beam side. Defaults to n_BG_in.
* `polarization`: `str: 's' or 'p'` or  `Polarization` class enum — incoming polarization. Defaults to p-polarization.

The function returns a named tuple called `TMMResult` with the following attributes, in order:
* `wavelengths` and `angles` (same as above) 
* `r`, `t`, `R`, `T`, `A` — respectively the field reflectivity and transmission, and the intensity reflectivity, 
transmission and absorption. These are each arrays of size `(len(wavelengths), len(angles_list))`

For `indices`, you typically want a repeating bi-layer of 2 materials. This can 
be created using the function `generate_index_bilayer()`, for example:

    from materials import n_Si, n_SiO2
    indices = generate_index_bilayer(n_Si, n_SiO2, wavelengths, num_layers)

### TMM()

`TMM()` is the pure numpy version of the `TMM()` function. It is entirely
vectorized and so is fairly quick despite not being jitted. It starts up very 
quickly with no lag and is a good baseline TMM function. However, if you're
planning on running the function 100s-1000s of times (such as in an 
optimization), the jitted version `TMM_jit()` of the function is way faster. 
Plus, you get autograd for free, which is amazing for gradient descent.

### TMM_jax()

`TMM_jax()` is an un-jitted version of the `TMM()` function. Due to the overhead
introduced by JAX, it's way way slower than either other version.

### TMM_jit()

`TMM_jit()` is a jitted version of `TMM_jax()`. The compilation time of jitting
makes this slower ot start up, but it's worth it if `TMM()` is being run many times.

### calc_M_matrix()
For certain applications, you might be interested in calculating the 
characteristic M-matrix directly. For this reason, we provide the calc_M_matrix().
For a jitted version you can access `calc_M_matrix_jax()` or `calc_M_matrix_jit()`.

Its input arguments are identical to `TMM()` or `TMM_uncompiled()`. It returns 
a complex array of size `(2, 2, len(wavelengths), len(angles_list))`.
