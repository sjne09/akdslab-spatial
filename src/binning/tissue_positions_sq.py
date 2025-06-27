import os
from math import ceil, cos, pi

import numpy as np
import pandas as pd
import shapely
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from tifffile import RESUNIT, TiffFile


def get_square_grid(s: float, h: int, w: int):
    """
    Parameters
    ----------
    s : float
        The side length of the square spots, in pixels. For Visium HD
        this should correspond to 2 um.
    """
    spots = []
    for b in range(w, step=2):
        for a in range(w, step=2):
            y = b % 2
            x = a % 2
            spots.append((y, x, b, a))
            print(y, x, b, a)

    return spots


def contains_tissue(
    O: tuple[int, int],  # noqa: E741
    s: float,
    img: np.ndarray,
    thresh: float,
) -> bool:
    """
    Determine if spot contains tissue using Otsu's method.

    Parameters
    ----------
    O : tuple[int, int]
        The top-left origin for the spot, in pixels

    d : int
        The side length of the spot, in pixels

    img : np.ndarray
        The grayscale image to check for tissue within the spot;
        shape [h, w]

    thresh : float
        The luminance threshold for distinguishing foreground from
        background

    Returns
    -------
    bool
        Whether a spot contains tissue
    """
    if len(img.shape) != 2:
        raise ValueError("`img` must be a grayscale image")

    a, b = O

    # crop the image to the bounds of the spot
    cropped_img = img[int(b) : int(b) + s + 1, int(a) : int(a) + s + 1]

    avg_luminance = cropped_img.mean()
    return avg_luminance < thresh


def get_tissue_positions(
    s: float, img: np.ndarray, out_dir: str
) -> pd.DataFrame:
    """
    Parameters
    ----------
    s : float
        The side length of the spots, in pixels

    distance : float
        The distance between spot centers, in pixels

    img : np.ndarray
        The full resolution image to build the matrix from;
        shape [h, w, c] where c is the number of channels
        (e.g. 3 for RGB)

    Returns
    -------
    pd.DataFrame
        A dataframe containing the tissue positions matrix
    """
    spots = get_square_grid(s, *img.shape[:-1])

    # get the luminance threshold for distinguishing f/g from b/g
    gscale = rgb2gray(img)
    thresh = threshold_otsu(gscale)

    positions = np.zeros((len(spots), 5), dtype=np.int32)
    for i, (y, x, b, a) in enumerate(spots):
        positions[i] = [
            int(contains_tissue((a, b), s, gscale, thresh)),
            y,
            x,
            b,
            a,
        ]

    # structure into dataframe to save as csv
    df = pd.DataFrame(
        positions,
        columns=[
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ],
    )
    df.to_csv(f"{out_dir}/tissue_positions.csv")
    return df
