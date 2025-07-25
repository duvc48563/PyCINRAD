import warnings

try:
    import cupy as cp
except Exception as e:
    cp = None
    _cupy_import_error = e


def array_to_rgba(data, cmap, norm):
    """Map data to an RGBA image using GPU via CuPy."""
    if cp is None:
        raise ImportError(
            "CuPy is required for GPU drawing but could not be imported"
        )
    cp_data = cp.asarray(data)
    normed = (cp_data - norm.vmin) / (norm.vmax - norm.vmin)
    normed = cp.clip(normed, 0.0, 1.0)
    indices = cp.floor(normed * (cmap.N - 1)).astype(cp.int32)
    lut = cmap(cp.linspace(0, 1, cmap.N))
    cp_lut = cp.asarray(lut)
    rgba = cp_lut[indices]
    return cp.asnumpy(rgba)
