import numpy as np
from einops import rearrange


def rearrange_for_filter(
    x: np.ndarray, axis: int
) -> tuple[np.ndarray, str, str, dict[str, int]]:
    """
    Rearranges an n-dimensional array for filtering along a specified
    axis. Places the specified axis first and flattens the other axes.
    Example: if x is a 3D array with shape (H, W, C) and axis=1,
    the output will be a 2D array with shape (W, H*C).

    Parameters
    ----------
    x : np.ndarray
        The input array to rearrange

    axis : int
        The axis to move to the front

    Returns
    -------
    tuple[np.ndarray, str, str, dict[str, int]]
        The rearranged array, the pattern for the pre-axis,
        the pattern for the post-axis, and a dictionary mapping the
        post-axis names to their sizes (for use in undoing the
        rearrangement)
    """
    convert_to_alpha = np.vectorize(lambda x: chr(x + 97))

    pattern_pre = np.arange(x.ndim)
    pattern_post = np.delete(pattern_pre, axis)
    axis_sizes = np.array(x.shape)[pattern_post]

    pattern_pre = " ".join(convert_to_alpha(pattern_pre))
    pattern_post = convert_to_alpha(pattern_post)
    axis_sizes = dict(zip(pattern_post, axis_sizes))

    pattern_post = convert_to_alpha(axis) + " (" + " ".join(pattern_post) + ")"

    return (
        rearrange(x, pattern_pre + " -> " + pattern_post),
        pattern_pre,
        pattern_post,
        axis_sizes,
    )


def filter1d(x: np.ndarray, size: int, axis: int) -> np.ndarray:
    """
    Applies a 1D filter to an n-dimensional array along a specified
    axis. Padding is applied using reflection to avoid edge effects.
    Ignores NaN values in the input array.

    Parameters
    ----------
    x : np.ndarray
        The input array to filter

    size : int
        The size of the filter window. Must be an odd integer

    axis : int
        The axis along which to apply the filter

    Returns
    -------
    np.ndarray
        The filtered array, with the same shape as the input array
    """
    x, pattern_pre, pattern_post, axis_sizes = rearrange_for_filter(
        x, axis=axis
    )

    H, W = x.shape
    output = np.zeros_like(x)

    reflection = (size - 1) // 2
    tmp = np.zeros((H + reflection * 2, W))
    tmp[reflection:-reflection, :] = x
    tmp[:reflection, :] = tmp[reflection * 2 - 1 : reflection - 1 : -1, :]
    tmp[-reflection:, :] = tmp[-reflection - 1 : -reflection * 2 - 1 : -1, :]
    x = tmp

    for j in range(W):
        windows = np.lib.stride_tricks.sliding_window_view(x[:, j], size)
        avgs = np.nanmean(windows, axis=1)
        output[:, j] = avgs

    # reverse the rearrangement before returning
    return rearrange(output, pattern_post + " -> " + pattern_pre, **axis_sizes)


def filter(x: np.ndarray, size: int) -> np.ndarray:
    """
    Applies a 1D filter to an n-dimensional array along each axis.
    Padding is applied using reflection to avoid edge effects.
    Effectively returns an array where each element is the mean of the
    elements in a window of size `size` centered on that element in
    `x`.

    Parameters
    ----------
    x : np.ndarray
        The input array to filter

    size : int
        The size of the filter window. Must be an odd integer

    Returns
    -------
    np.ndarray
        The filtered array, with the same shape as the input array
    """
    for axis in range(x.ndim):
        output = filter1d(x=x, size=size, axis=axis)
        x = output
    return x


def structural_similarity(
    im1: np.ndarray, im2: np.ndarray, size: int = 7, data_range: float = 1.0
) -> tuple[float, np.ndarray]:
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    Ignores NaN values in the input images.

    Parameters
    ----------
    im1 : np.ndarray
        The first image to compare

    im2 : np.ndarray
        The second image to compare

    size : int, optional
        The size of the filter window. Must be an odd integer.
        Default is 7

    data_range : float, optional
        The dynamic range of the images. Default is 1.0

    Returns
    -------
    tuple[float, np.ndarray]
        The mean SSIM value and the SSIM map between the two images.
        The SSIM map is a 2D array where each element corresponds to
        the SSIM value for the corresponding pixel in the input
        images
    """
    if np.any((np.asarray(im1.shape) - size) < 0):
        raise ValueError(
            "win_size exceeds image extent. "
            "Either ensure that your images are "
            f"at least {size}x{size}; or pass win_size explicitly "
            "in the function call, with an odd value "
            "less than or equal to the smaller side of your "
            "images. If your images are multichannel "
            "(with color channels), set channel_axis to "
            "the axis number corresponding to the channels."
        )

    if not (size % 2 == 1):
        raise ValueError("Window size must be odd.")

    # ensure float32
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # filter is normalized by NP, so must first mult by NP, then divide
    # by NP - 1 for sample covar
    ndim = im1.ndim
    NP = size**ndim
    cov_norm = NP / (NP - 1)

    ux = filter(im1, size=size)
    uy = filter(im2, size=size)

    # compute (weighted) variances and covariances
    uxx = filter(im1 * im1, size=size)
    uyy = filter(im2 * im2, size=size)
    uxy = filter(im1 * im2, size=size)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    L = data_range
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (size - 1) // 2
    mssim = S[pad:-pad, pad:-pad].mean(dtype=np.float64)
    return mssim, S
