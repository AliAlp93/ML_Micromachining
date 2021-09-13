"""
Common-use utility functions which are reused across the project.
"""
from typing import cast
import numpy as np
from scipy import interpolate as interp  # type: ignore


def resample_vector(v: np.ndarray, targ_size: int, kind: str = 'linear') -> np.ndarray:
    """
    Resamples the given vector `v` using `kind` interpolation (e.g. linear) so 
    that the returned vector has `targ_size` entries.

    Equivalent to up- or down-sampling depending on whether `targ_size` is > or 
    < than `v.size`, respectively.

    Used for standardizing the length of vectors produced using different sample 
    rates.

    The output will be reshaped to match the shape of the input, that is data 
    will be along the same axis. E.g.: If `v` has shape `(1,30,1)` and is being 
    upsampled to `50` points, the output will be `(1,50,1)`.

    Args:
        v (np.ndarray): Vector (1D) to resample.
        targ_size (int): Target size for return vector.

    Returns:
        [np.ndarray]: Vector `v` resampled to have size `targ_size`.
    """
    # Check if y is 1D (along any axis):
    if v.size != max(v.shape):
        raise ValueError(
            f"Given vector `v={v}` must be 1D along any axis "
            "(it can be `(1,n,1)` for example but not `(2,n)`). "
            f"Instead, `v` has {v.size} elements in shape `{v.shape}`."
        )

    orig_shape = v.shape
    v = v.flatten()

    x_old = np.arange(v.size)
    x_new = np.linspace(0, v.size-1, targ_size)
    v_new = interp.interp1d(x_old, v, kind=kind)(x_new)

    out_shape = [*orig_shape]
    out_shape[int(np.argmax(orig_shape))] = targ_size

    return cast(np.ndarray, v_new.reshape(out_shape))
