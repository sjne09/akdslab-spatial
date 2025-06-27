import os
from math import ceil, cos, pi

import numpy as np
import pandas as pd
import shapely
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from tifffile import RESUNIT, TiffFile

from src.binning.utils import approx_circle

SCALE_FACTORS = {
    RESUNIT.INCH: 25.4,
    RESUNIT.CENTIMETER: 1.0e4,
    RESUNIT.MILLIMETER: 1.0e3,
    RESUNIT.MICROMETER: 1.0,
    RESUNIT.NONE: 1.0,
}


def get_hex_grid(
    d: float, distance: float, h: int, w: int
) -> list[tuple[int, int, int, int]]:
    """
    Returns the hex lattice spot grid that fills a space with shape
    [`h`, `w`] with spots defined by `d` and `distance`.

    Parameters
    ----------
    d : float
        The diameter of the spots, in pixels. For Visium this should
        correspond to 55 um

    distance : float
        The distance between spot centers, in pixels. For Visium this
        should correspond to 100 um

    h : int
        The height of the image, in pixels

    w : int
        The width of the image, in pixels

    Returns
    -------
    list[tuple[int, int, int, int]]
        array_row, array_col, pxl_row, pxl_col
    """
    # the first spot is centered such that its edges touch the boundaries of
    # the image, so O = (r, r)
    a_0 = ceil(d / 2)
    b_0 = ceil(d / 2)

    # the number of rows and cols of spots are defined in terms of the input
    # params
    # FIXME maybe int not necessary?
    i_max = int(h / (distance * cos(pi / 6)))
    j_max = int(w / distance)

    # fig, ax = plt.subplots(figsize=(10, 10))

    spots = []
    for i in range(i_max):
        # x progresses in multiples of 2 starting from 0 if i is even;
        # multiples of 2 starting from 1 if i is odd
        x_0 = 0 if (i % 2 == 0) else 1

        for j in range(j_max):
            # a, b are the spatial coords of the spot centers on the slide
            a = a_0 + int((distance * j) + ((i % 2) * distance / 2))
            b = b_0 + int((distance * cos(pi / 6) * i))

            # x, y are the array coords of the spots
            x = x_0 + j * 2
            y = i
            spots.append((y, x, b, a))
            # circle = plt.Circle((a, b), radius=55 / 2, color="r")
            # ax.add_patch(circle)

    # ax.set_xlim(0, 6500)
    # ax.set_ylim(0, 6500)
    # plt.savefig("test.png")

    return spots


def contains_tissue(
    O: tuple[int, int],  # noqa: E741
    d: float,
    img: np.ndarray,
    thresh: float,
) -> bool:
    """
    Determine if spot contains tissue using Otsu's method.

    Parameters
    ----------
    O : tuple[int, int]
        The spot center slide coordinates as (x, y)

    d : int
        The diameter of the spot, in pixels

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

    # approximate the circle
    circle = approx_circle(a, b, d / 2)
    minx, miny, maxx, maxy = circle.bounds

    # crop the image to the bounds of the circle
    cropped_img = img[int(miny) : int(maxy) + 1, int(minx) : int(maxx) + 1]

    # create a mask over the cropped image for pixels within the circle;
    # adjust the point for `contains` to the original image coords using
    # the circle's minx and miny
    mask = np.array(
        [
            [
                circle.contains(shapely.Point(x + int(minx), y + int(miny)))
                for x in range(cropped_img.shape[1])
            ]
            for y in range(cropped_img.shape[0])
        ]
    )

    # get the avg luminance based on pixels within the circle
    aoi = cropped_img[mask]
    avg_luminance = aoi.mean()
    return avg_luminance < thresh


def get_tissue_positions(
    d: float, distance: float, img: np.ndarray, out_dir: str
) -> pd.DataFrame:
    """
    Constructs a tissue positions matrix, replicating output from
    Visium. The matrix will be saved to a csv called
    "tissue_positions.csv" and contain the following columns:
    in_tissue, array_row, array_col, pxl_row_in_fullres,
    pxl_col_in_fullres.

    Parameters
    ----------
    d : float
        The diameter of the spots, in pixels

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
    spots = get_hex_grid(d, distance, *img.shape[:-1])

    # get the luminance threshold for distinguishing f/g from b/g
    gscale = rgb2gray(img)
    thresh = threshold_otsu(gscale)

    positions = np.zeros((len(spots), 5), dtype=np.int32)
    for i, (y, x, b, a) in enumerate(spots):
        positions[i] = [
            int(contains_tissue((a, b), d, gscale, thresh)),
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


def locs_from_tissue_positions(
    tissue_positions: pd.DataFrame, out_dir: str
) -> None:
    """
    Converts a tissue positions matrix into the locs matrix expected
    by iStar.

    Parameters
    ----------
    tissue_positions : pd.DataFrame
        The tissue positions matrix
    """
    df = tissue_positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].copy()
    df.rename(
        columns={
            "pxl_col_in_fullres": "x",
            "pxl_row_in_fullres": "y",
        },
        inplace=True,
    )
    df["spot"] = (
        tissue_positions["array_row"].astype(str)
        + "x"
        + tissue_positions["array_col"].astype(str)
    )
    df = (
        df[["spot", "x", "y"]]
        .where(tissue_positions["in_tissue"] == 1)
        .dropna()
    )
    df.to_csv(f"{out_dir}/locs-raw.tsv", index=False, sep="\t")


def visualize(
    tissue_positions: pd.DataFrame,
    img: str | TiffFile,
    out_dir: str,
    radius: float,
    max_width: int = 2000,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image

    if isinstance(img, str):
        img = TiffFile(img)

    img_array = img.asarray(0)
    pil_img = Image.fromarray(img_array)

    h, w = img_array.shape[:-1]
    aspect_ratio = w / h
    scale_ratio = max_width / w
    max_height = max_width / aspect_ratio

    # rescale image
    pil_img.thumbnail((max_width, max_height))

    # modify radius
    rescaled_radius = radius * scale_ratio

    # rescale_coords
    tissue_positions["pxl_row_rescaled"] = (
        (tissue_positions["pxl_row_in_fullres"] * scale_ratio)
        .round(0)
        .astype(int)
    )
    tissue_positions["pxl_col_rescaled"] = (
        (tissue_positions["pxl_col_in_fullres"] * scale_ratio)
        .round(0)
        .astype(int)
    )

    # plot
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pil_img)

    coords = tissue_positions[
        ["pxl_col_rescaled", "pxl_row_rescaled", "in_tissue"]
    ].to_numpy()

    for x, y, it in coords:
        c = Circle(
            (x, y),
            facecolor="blue",
            alpha=1 if it == 1 else 0,  # set opacity to 0 if not in tissue
            radius=rescaled_radius,
        )
        ax.add_patch(c)

    plt.savefig(f"{out_dir}/vis.png", dpi=300)


def get_radius_in_px(mpp: float, r_in_um: float, out_dir: str) -> float:
    r_in_px = r_in_um / mpp

    with open(f"{out_dir}/radius-raw.txt", "w") as f:
        f.write(str(r_in_px))

    return r_in_px


def get_mpp(tif: TiffFile, out_dir: str) -> float:
    # get the scale factor to convert px/unit to um/px
    scalef = SCALE_FACTORS[tif.pages[0].tags["ResolutionUnit"].value]

    # XResolution is px per units in a (px, units) tuple
    mpp = (
        tif.pages[0].tags["XResolution"].value[1]  # units
        / tif.pages[0].tags["XResolution"].value[0]  # px
        * scalef
    )

    # correct for errors in image metadata
    if mpp < 0.1:
        mpp *= 10

    with open(f"{out_dir}/pixel-size-raw.txt", "w") as f:
        f.write(str(mpp))

    return mpp


if __name__ == "__main__":
    sample_ids = ["TENX111", "TENX114", "TENX147", "TENX148", "TENX149"]
    excl = {"TENX111", "TENX114"}

    for sample_id in sample_ids:
        if sample_id in excl:
            print(f"Skipping {sample_id}")
            continue

        print(f"Processing {sample_id}...")
        os.makedirs(sample_id, exist_ok=True)

        with TiffFile(
            f"/opt/gpudata/sjne/HEST/data/wsis/{sample_id}.tif"
        ) as slide:
            d_in_um = 55

            um_per_px = get_mpp(slide, sample_id)
            r_in_px = get_radius_in_px(um_per_px, d_in_um / 2, sample_id)

            distance_in_um = 100
            distance_in_px = distance_in_um / um_per_px

            print("Getting tissue positions")
            tp = get_tissue_positions(
                r_in_px * 2, distance_in_px, slide.asarray(0), sample_id
            )

            print("Building locs dataframe")
            locs_from_tissue_positions(tp, sample_id)

            tp = pd.read_csv(f"{sample_id}/tissue_positions.csv")

            try:
                print("Visualizing")
                visualize(tp, slide, sample_id, r_in_px)
            except Exception:
                pass
