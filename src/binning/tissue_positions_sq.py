import os

import cudf
import cupy as cp
from cucim.skimage.filters import threshold_otsu
from PIL import Image
from skimage.color import rgb2gray  # cpu to avoid OOM for large images
from tifffile import TiffFile

from src.binning.utils import get_mpp, locs_from_tissue_positions


def get_side_in_px(mpp: float, s_in_um: float, out_dir: str) -> int:
    """
    Converts the side length of the square spots from microns to
    pixels based on the microns per pixel (mpp) value.

    Parameters
    ----------
    mpp : float
        The microns per pixel value of the HE image

    s_in_um : float
        The side length of the square spots in microns

    out_dir : str
        The directory to save the side length value to

    Returns
    -------
    int
        The side length of the square spots in pixels
    """
    s_in_px = int(s_in_um / mpp)

    with open(f"{out_dir}/side-raw.txt", "w") as f:
        f.write(str(s_in_px))

    return s_in_px


def get_square_grid(s: int, h: int, w: int) -> list[tuple[int, int, int, int]]:
    """
    Returns the square grid that fills a space with shape [`h`, `w`]
    with spots defined by `s`.

    Parameters
    ----------
    s : int
        The side length of the square spots, in pixels

    h : int
        The height of the image, in pixels

    w : int
        The width of the image, in pixels

    Returns
    -------
    list[tuple[int, int, int, int]]
        array_row, array_col, pxl_row, pxl_col
    """
    spots = []
    for b in range(0, h, s):
        for a in range(0, w, s):
            y = b // s
            x = a // s
            spots.append((y, x, b, a))

    return spots


def contains_tissue(
    O: tuple[int, int],  # noqa: E741
    s: int,
    img: cp.ndarray,
    thresh: float,
) -> int:
    """
    Determine if a spot contains tissue using Otsu's method.

    Parameters
    ----------
    O : tuple[int, int]
        The top-left origin for the spot, in pixels

    s : int
        The side length of the spot, in pixels

    img : cp.ndarray
        The grayscale image to check for tissue within the spot;
        shape [h, w]

    thresh : float
        The luminance threshold for distinguishing foreground from
        background

    Returns
    -------
    int
        Binary indicator of whether a spot contains tissue. 1 if it
        does, 0 otherwise
    """
    if len(img.shape) != 2:
        raise ValueError("`img` must be a grayscale image")

    x, y = O

    # crop the image to the bounds of the spot
    cropped_img = img[
        int(y) : min(int(y) + s, img.shape[0]),
        int(x) : min(int(x) + s, img.shape[1]),
    ]

    # check if luminance above threshold
    avg_luminance = cropped_img.mean()
    return int(avg_luminance < thresh)


def get_tissue_positions(
    s: int, img: cp.ndarray, out_dir: str
) -> cudf.DataFrame:
    """
    Constructs a tissue positions matrix, replicating output from
    Visium. The matrix will be saved to a csv called
    "tissue_positions.csv" and contain the following columns:
    in_tissue, array_row, array_col, pxl_row_in_fullres,
    pxl_col_in_fullres.

    Parameters
    ----------
    s : int
        The side length of the spots, in pixels

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
    spots = get_square_grid(s, *img.shape[:-1])
    print("Grid created; calculating tissue flags")

    # get the luminance threshold for distinguishing f/g from b/g
    img_cpu = cp.asnumpy(img)  # send img to cpu; delete from gpu
    del img

    gscale = rgb2gray(img_cpu)
    gscale = cp.array(gscale)  # move gscale to gpu
    thresh = threshold_otsu(gscale)

    # identify spots that contain tissue and form positions array
    positions = cp.zeros((len(spots), 5), dtype=cp.int32)
    for i, (arr_row, arr_col, pxl_row, pxl_col) in enumerate(spots):
        tissue_flag = contains_tissue((pxl_col, pxl_row), s, gscale, thresh)
        positions[i] = cp.array(
            [tissue_flag, arr_row, arr_col, pxl_row, pxl_col]
        )

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
    s: int,
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

    s : int
        The side length of the square spots, in pixels

    max_width : int, optional
        The maximum width of the output image, by default 2000
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
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

    # modify side length
    rescaled_s = s * scale_ratio

    # rescale_coords
    tissue_positions["pxl_row_rescaled"] = (
        tissue_positions["array_row"] * rescaled_s
    ).astype(int)
    tissue_positions["pxl_col_rescaled"] = (
        tissue_positions["array_col"] * rescaled_s
    ).astype(int)

    # plot
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pil_img)

    coords = tissue_positions[
        ["pxl_col_rescaled", "pxl_row_rescaled", "in_tissue"]
    ].to_numpy()

    for x, y, it in coords:
        sq = Rectangle(
            (x, y),
            height=rescaled_s,
            width=rescaled_s,
            fill=None,
            edgecolor="blue",
            linewidth=0.2,
            alpha=1 if it == 1 else 0,  # set opacity to 0 if not in tissue
        )
        ax.add_patch(sq)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{out_dir}/vis.png", dpi=300, bbox_inches="tight")
    plt.close()


def run(s: int, img_path: str, out_dir: str) -> None:
    """
    Runs the tissue positions extraction and visualization for square
    grid spots.

    Parameters
    ----------
    s : int
        The side length of the square spots, in microns/um

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

        um_per_px = get_mpp(slide, out_dir)
        s_in_px = get_side_in_px(um_per_px, s, out_dir)

        print("Getting tissue positions")
        tp = get_tissue_positions(
            s_in_px,
            cp.asarray(slide.asarray(0)),
            out_dir,
        )

        print("Building locs dataframe")
        locs_from_tissue_positions(tp, out_dir)

        try:
            print("Visualizing")
            visualize(tp, slide, out_dir, s_in_px)
        except Exception:
            pass


if __name__ == "__main__":
    sample_ids = ["TENX111", "TENX114", "TENX147", "TENX148", "TENX149"]
    excl = {"TENX114", "TENX147", "TENX148", "TENX149"}

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
            um_per_px = get_mpp(slide, sample_id)
            s_in_px = int(2 / um_per_px)

            # print("Getting tissue positions")
            tp = get_tissue_positions(
                s_in_px,
                cp.asarray(slide.asarray(0)),
                sample_id,
            )

            tp = cudf.read_csv(
                "/home/sjne/projects/akdslab-spatial/TENX111/tissue_positions_sq_7.csv"
            )

            print("Building locs dataframe")
            locs_from_tissue_positions(tp, sample_id)

            # try:
            print("Visualizing")
            visualize(tp, slide, sample_id, s_in_px)
            # except Exception:
            #     pass
