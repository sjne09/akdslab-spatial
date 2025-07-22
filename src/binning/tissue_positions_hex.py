import os
from math import ceil, cos, pi

import cudf
import cupy as cp
from cucim.skimage.filters import threshold_otsu
from PIL import Image
from skimage.color import rgb2gray  # cpu to avoid OOM for large images
from tifffile import TiffFile

from src.binning.utils import (
    get_mpp,
    locs_from_tissue_positions,
)


def get_radius_in_px(mpp: float, r_in_um: float, out_dir: str) -> float:
    """
    Converts the radius from microns to pixels based on the
    microns-per-pixel (mpp) value.

    Parameters
    ----------
    mpp : float
        Microns per pixel value for the image

    r_in_um : float
        Radius in microns

    out_dir : str
        The directory to save the radius value to

    Returns
    -------
    float
        The radius in pixels
    """
    r_in_px = r_in_um / mpp

    with open(f"{out_dir}/radius-raw.txt", "w") as f:
        f.write(str(r_in_px))

    return r_in_px


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
    i_max = int(h / (distance * cos(pi / 6)))
    j_max = int(w / distance)

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

    return spots


def contains_tissue(
    O: tuple[int, int],  # noqa: E741
    r: float,
    img: cp.ndarray,
    thresh: float,
) -> int:
    """
    Determine if a spot contains tissue using Otsu's method.

    Parameters
    ----------
    O : tuple[int, int]
        The coordinates for the center of the spot, in pixels

    r : float
        The radius of the spot, in pixels

    img : cp.ndarray
        The grayscale image to check for tissue within the spot

    thresh : float
        The luminance threshold for distinguishing foreground from
        background

    Returns
    -------
    int
        Binary indicator of whether a spot contains tissue. 1 if it
        does, 0 otherwise
    """
    a, b = O
    h, w = img.shape

    # bbox for spot
    x_min = max(0, int(a - r - 1))
    x_max = min(w, int(a + r + 2))
    y_min = max(0, int(b - r - 1))
    y_max = min(h, int(b + r + 2))

    # get all possible points within spot bbox (cartesian prod)
    y_coords, x_coords = cp.meshgrid(
        cp.arange(y_min, y_max), cp.arange(x_min, x_max), indexing="ij"
    )

    # calc distances between each point and the circle center
    distances = cp.sqrt(cp.power(x_coords - a, 2) + cp.power(y_coords - b, 2))

    # create a mask to identify points within spot circle
    mask = distances <= r

    # get avg luminance within grayscale patch; set tissue based on lum.
    # and threshold
    if cp.any(mask):
        region = img[y_min:y_max, x_min:x_max]
        avg_luminance = region[mask].mean()
        return int(avg_luminance < thresh)


def get_tissue_positions(
    d: float, distance: float, img: cp.ndarray, out_dir: str
) -> cudf.DataFrame:
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

    img : cp.ndarray
        The full resolution image to build the matrix from;
        shape [h, w, c] where c is the number of channels
        (e.g. 3 for RGB)

    out_dir : str
        The directory to save the output to

    Returns
    -------
    cudf.DataFrame
        A dataframe containing the tissue positions matrix
    """
    spots = get_hex_grid(d, distance, *img.shape[:-1])
    print("Hex grid created; calculating tissue flags")

    # get the luminance threshold for distinguishing f/g from b/g
    img_cpu = cp.asnumpy(img)  # send img to cpu; delete from gpu
    del img

    gscale = rgb2gray(img_cpu)
    gscale = cp.array(gscale)  # move gscale to gpu
    thresh = threshold_otsu(gscale)

    # identify spots that contain tissue and form positions array
    positions = cp.zeros((len(spots), 5), dtype=cp.int32)
    for i, (y, x, b, a) in enumerate(spots):
        tissue_flag = contains_tissue((a, b), d / 2, gscale, thresh)
        positions[i] = cp.array([tissue_flag, y, x, b, a])

    # structure into dataframe to save as csv
    df = cudf.DataFrame(
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


def visualize(
    tissue_positions: cudf.DataFrame,
    img: str | TiffFile,
    out_dir: str,
    radius: float,
    max_width: int = 2000,
) -> None:
    """
    Visualizes the tissue positions on the HE image, saving the output
    as a PNG file.

    Parameters
    ----------
    tissue_positions : cudf.DataFrame
        A DataFrame containing the tissue positions matrix with columns
        "in_tissue", "array_row", "array_col", "pxl_row_in_fullres",
        "pxl_col_in_fullres"

    img : str | TiffFile
        The path to the HE image or a TiffFile object containing the HE
        image data

    out_dir : str
        The directory to save the visualization to

    radius : float
        The radius of the spots in pixels

    max_width : int, optional
        The maximum width of the output image, by default 2000
    """
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

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{out_dir}/vis.png", dpi=300, bbox_inches="tight")
    plt.close()


def run(d: float, dist: float, img_path: str, out_dir: str) -> None:
    """
    Runs the tissue positions extraction and visualization for
    hexagonal grids.

    Parameters
    ----------
    d : float
        The diameter of the spots, in microns/um. For Visium this is 55

    dist : float
        The distance between spot centers, in microns/um. For Visium
        this is 100

    img_path : str
        The path to the HE image file

    out_dir : str
        The directory to save the output files to
    """
    import gc

    Image.MAX_IMAGE_PIXELS = None

    with TiffFile(img_path) as slide:
        img = Image.fromarray(slide.asarray(0))
        img.save(f"{out_dir}/he-raw.png")
        del img
        gc.collect()

        # convert d and dist to pixels
        d_in_um = d

        um_per_px = get_mpp(slide, out_dir)
        r_in_px = get_radius_in_px(um_per_px, d_in_um / 2, out_dir)

        distance_in_um = dist
        distance_in_px = distance_in_um / um_per_px

        # generate the tissue positions dataframe
        print("Getting tissue positions")
        tp = get_tissue_positions(
            r_in_px * 2,
            distance_in_px,
            cp.asarray(slide.asarray(0)),
            out_dir,
        )

        # generate the locs dataframe required for iSTAR
        print("Building locs dataframe")
        locs_from_tissue_positions(tp, out_dir)

        # visualize the spots on the HE image
        try:
            print("Visualizing")
            visualize(tp, slide, out_dir, r_in_px)
        except Exception:
            pass


if __name__ == "__main__":
    sample_ids = ["TENX111", "TENX114", "TENX147", "TENX148", "TENX149"]
    # excl = {"TENX114", "TENX147", "TENX148", "TENX149"}
    excl = {}

    Image.MAX_IMAGE_PIXELS = None

    for sample_id in sample_ids:
        if sample_id in excl:
            print(f"Skipping {sample_id}")
            continue

        print(f"Processing {sample_id}...")
        os.makedirs(sample_id, exist_ok=True)

        with TiffFile(
            f"/opt/gpudata/sjne/HEST/data/wsis/{sample_id}.tif"
        ) as slide:
            img = Image.fromarray(slide.asarray(0))
            img.save(f"{sample_id}/he-raw.png")

            d_in_um = 55

            um_per_px = get_mpp(slide, sample_id)
            r_in_px = get_radius_in_px(um_per_px, d_in_um / 2, sample_id)

            distance_in_um = 100
            distance_in_px = distance_in_um / um_per_px

            print("Getting tissue positions")
            tp = get_tissue_positions(
                r_in_px * 2,
                distance_in_px,
                cp.asarray(slide.asarray(0)),
                sample_id,
            )

            print("Building locs dataframe")
            locs_from_tissue_positions(tp, sample_id)

            # tp = cudf.read_csv(f"{sample_id}/tissue_positions.csv")

            try:
                print("Visualizing")
                visualize(tp, slide, sample_id, r_in_px)
            except Exception:
                pass
