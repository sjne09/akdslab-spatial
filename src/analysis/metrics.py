import pickle
import warnings
from dataclasses import dataclass
from math import sqrt
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from src.analysis.ssim import structural_similarity

# suppress warnings about mean of empty slice (from np.nanmean)
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)


@dataclass
class Counts:
    """
    Dataclass to hold gene expression counts and gene names.

    Attributes
    ----------
    counts : np.ndarray
        A 3D numpy array of shape (n_genes, n_rows, n_cols) containing
        the expression counts for each gene across all spots

    genes : np.ndarray
        A 1D numpy array of shape (n_genes,) containing the names of the
        genes corresponding to the counts in the `counts` attribute
    """

    counts: np.ndarray  # int array
    genes: np.ndarray  # string array


@dataclass
class BoundingBox:
    """
    Dataclass to define a bounding box for cropping gene expression
    counts data.

    Attributes
    ----------
    min_r : int
        The minimum row index of the bounding box

    max_r : int
        The maximum row index of the bounding box

    min_c : int
        The minimum column index of the bounding box

    max_c : int
        The maximum column index of the bounding box
    """

    min_r: int
    max_r: int
    min_c: int
    max_c: int


def load_data(
    dir_path: Path,
    normalize: bool,
    mask: None | np.ndarray = None,
    bbox: None | BoundingBox = None,
) -> Counts:
    """
    Loads gene expression data from a directory containing pickled
    files for each gene. The files are expected to contain a
    2-dimentional numpy array with expression counts.

    Parameters
    ----------
    dir_path : Path
        Path to the directory containing the gene expression data files

    normalize : bool
        Whether to apply min-max normalization to the counts data.
        Normalization ignores NaN values

    mask : None | np.ndarray
        A mask for the original HE image defining the tissue limits.
        The shape of the mask must match the shape of the counts data.
        If provided, the counts data will be masked to only include
        values within the mask. Values outside the mask will be set to
        NaN

    bbox : None | BoundingBox
        A bounding box defining the region of interest in the counts
        data. If provided, the counts data will be cropped to this
        bounding box

    Returns
    -------
    Counts
        An instance of the Counts dataclass containing the counts
        array and the gene names. The counts array will have a shape
        of (n_genes, n_rows, n_cols), where n_genes is the number of
        genes - the order of which matches the order of the genes in
        the `genes` attribute
    """
    full_counts = []
    genes = []
    for gene_data in dir_path.iterdir():
        genes.append(gene_data.stem)

        with open(gene_data, "rb") as f:
            counts_by_spot = pickle.load(f)

        # apply mask if provided
        if mask is not None:
            counts_by_spot[~mask] = np.nan

        # crop to ROI if bbox is provided
        if bbox is not None:
            counts_by_spot = counts_by_spot[
                bbox.min_r : bbox.max_r + 1, bbox.min_c : bbox.max_c + 1
            ]

        # normalize last to ensure normalization only over remaining values
        if normalize:
            counts_by_spot = normalize(counts_by_spot)

        full_counts.append(counts_by_spot)

    genes = np.array(genes)
    full_counts = np.array(full_counts)

    return Counts(counts=full_counts, genes=genes)


def normalize(counts: np.ndarray) -> np.ndarray:
    """
    Applies min-max normalization to the counts data. Ignores NaN
    values.

    Parameters
    ----------
    counts : np.ndarray
        The counts matrix, which can be either 2D or 3D.
        If 2D, it is assumed to be a single gene's counts data

    Returns
    -------
    np.ndarray
        The normalized counts matrix, with the same shape as the input
        counts. The values are scaled to the range [0, 1]
    """
    if len(counts.shape) == 3:
        norm = np.zeros_like(counts)
        for i, gene in enumerate(counts):
            norm[i] = gene - np.nanmin(gene)
            norm[i] /= np.nanmax(norm[i]) + 1e-12
    else:
        norm = counts - np.nanmin(counts)
        norm /= np.nanmax(norm) + 1e-12
    return norm


def rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculates RMSE.

    Parameters
    ----------
    gt : np.ndarray
        The ground truth / actual values; must be the same shape as
        `pred`

    pred : np.ndarray
        The predicted values; must be the same shape as `gt`

    Returns
    -------
    float
        The RMSE
    """
    se = np.power(gt - pred, 2)
    mse = np.nanmean(se)
    _rmse = sqrt(mse)
    return _rmse


def mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculates MAE.

    Parameters
    ----------
    gt : np.ndarray
        The ground truth / actual values; must be the same shape as
        `pred`

    pred : np.ndarray
        The predicted values; must be the same shape as `gt`

    Returns
    -------
    float
        The MAE
    """
    ae = np.abs(gt - pred)
    _mae = np.nanmean(ae)
    return _mae


def normalize_with_shared_params(
    gt_counts: np.ndarray, pred_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize both gt and pred using shared min/max parameters.
    Uses the combined min/max from both datasets to ensure same scale.
    """
    gt_norm = np.zeros_like(gt_counts)
    pred_norm = np.zeros_like(pred_counts)

    for i in range(gt_counts.shape[0]):
        # get combined min/max for this gene
        combined_min = min(np.nanmin(gt_counts[i]), np.nanmin(pred_counts[i]))
        combined_max = max(np.nanmax(gt_counts[i]), np.nanmax(pred_counts[i]))

        # normalize gt and preds using the same parameters
        gt_norm[i] = (gt_counts[i] - combined_min) / (
            combined_max - combined_min + 1e-12
        )
        pred_norm[i] = (pred_counts[i] - combined_min) / (
            combined_max - combined_min + 1e-12
        )

    return gt_norm, pred_norm


def ssim(
    im1: np.ndarray,
    im2: np.ndarray,
    mask: np.ndarray,
    size: int = 7,
    data_range: float = 1.0,
) -> float:
    """
    Calculates the Structural Similarity Index (SSIM), ignoring NaN
    values.

    Parameters
    ----------
    im1 : np.ndarray
        The first image (ground truth) for comparison. Must be the same
        shape as `im2`

    im2 : np.ndarray
        The second image (predicted) for comparison. Must be the same
        shape as `im1`

    mask : np.ndarray
        A mask defining the valid regions for SSIM calculation.
        The shape of the mask must match the shape of the images

    size : int, optional
        The size of the sliding window for SSIM calculation. Default is
        7. Must be an odd integer that is less than or equal to the
        size of any dimension of the images

    data_range : float, optional
        The data range of the input images. Default is 1.0. This is
        used to normalize the pixel values for SSIM calculation

    Returns
    -------
    float
        The mean SSIM value for the regions defined by the mask
    """
    _, S = structural_similarity(im1, im2, size, data_range)
    S[~mask] = np.nan
    return np.nanmean(S)


def get_and_preprocess_data(
    data_dir: Path,
    mask: None | np.ndarray = None,
    bbox: None | BoundingBox = None,
) -> Counts:
    """
    Loads and preprocesses the gene expression data from the specified
    directory. The data is expected to be in a pickled format, with
    each gene's counts stored in a separate file.

    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the gene expression data files

    mask : None | np.ndarray
        A mask for the original HE image defining the tissue limits.
        The shape of the mask must correspond to the resolution of the
        gene expression data. If provided, the counts data will be
        masked to only include values within the mask. Values outside
        the mask will be set to NaN

    bbox : None | BoundingBox
        A bounding box defining the region of interest in the counts
        data. If provided, the counts data will be cropped to this
        bounding box

    Returns
    -------
    Counts
        An instance of the Counts dataclass containing the counts
        array and the gene names. The counts array will have a shape
        of (n_genes, n_rows, n_cols), where n_genes is the number of
        genes - the order of which matches the order of the genes in
        the `genes` attribute
    """
    data = load_data(data_dir, normalize=False, mask=mask, bbox=bbox)

    # get indices of gene names in alpha order
    genes_sorted_idx = np.argsort(data.genes)
    data.genes = data.genes[genes_sorted_idx]

    # apply gene sorting to counts matrix
    data.counts = data.counts[genes_sorted_idx]
    return data


def get_gt_bounds(counts_path: Path) -> BoundingBox:
    """
    Returns
    -------
    BoundingBox
        The extent of the ground truth data
    """
    spot_ids = pd.read_csv(counts_path, sep="\t", usecols=["spot_id"])[
        "spot_id"
    ]

    # get coords of spots in gt counts file and split
    coords = spot_ids.str.split("x", expand=True).astype(int)

    # get bounding box coords
    min_r = max(coords[0].min(), 0)
    max_r = coords[0].max()
    min_c = max(coords[1].min(), 0)
    max_c = coords[1].max()

    return BoundingBox(
        min_r=min_r,
        max_r=max_r,
        min_c=min_c,
        max_c=max_c,
    )


def visualize_roi(sample_dir: Path, bbox: BoundingBox) -> None:
    """
    Open the mask image, draw a bounding box around the region of
    interest specified by `bbox`, and save the result.

    Parameters
    ----------
    sample_dir : Path
        Directory containing the mask image

    bbox : BoundingBox
        Bounding box coordinates for the region of interest
    """
    mask_image_path = sample_dir / "mask-small.png"
    output_image_path = sample_dir / "cropped.jpg"

    Image.MAX_IMAGE_PIXELS = None

    with Image.open(mask_image_path) as img:
        # Convert to RGB if needed for drawing
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Create a drawing context
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)

        # Draw rectangle with no fill (outline only)
        draw.rectangle(
            [bbox.min_c, bbox.min_r, bbox.max_c, bbox.max_r],
            outline="red",
            width=2,
        )

        img.save(output_image_path)


def calculate_gene_metrics(args):
    """
    Calculate metrics for a single gene.

    Parameters
    ----------
    args : tuple
        Tuple containing (gene_index, gt_gene_data, pred_gene_data, mask)

    Returns
    -------
    tuple
        Tuple containing (gene_index, rmse_value, mae_value, ssim_value)
    """
    i, gt_gene, pred_gene, mask = args

    _rmse = rmse(gt_gene, pred_gene)
    _mae = mae(gt_gene, pred_gene)
    _ssim = ssim(gt_gene, pred_gene, mask=mask)

    return i, _rmse, _mae, _ssim


def main():
    base_dir = Path("/opt/gpudata/sjne/data_for_istar")
    dirs = [d for d in base_dir.iterdir() if d.stem.startswith("hex")]
    data_folder = "cnts-super"

    for d in dirs:
        logger.info(f"Beginning processing {d.name}")

        for sample_dir in d.iterdir():
            logger.info(f"Processing {sample_dir.name}")

            with Image.open(sample_dir / "mask-small.png") as img:
                mask = np.array(img)

            # load ground truth
            mpp = d.stem.rsplit("_", maxsplit=1)[-1]
            sample_id = sample_dir.stem
            gt_base_dir = base_dir / f"sq_{mpp}" / sample_id
            gt_dir = gt_base_dir / data_folder

            bbox = get_gt_bounds(gt_base_dir / "cnts.tsv")
            # crop_and_save_he_image(sample_dir, bbox)

            gt = get_and_preprocess_data(data_dir=gt_dir, bbox=bbox)

            # load preds
            preds_dir = sample_dir / data_folder
            preds = get_and_preprocess_data(
                data_dir=preds_dir, mask=mask, bbox=bbox
            )

            # control for the case where the number of genes is not equal
            # typically occurs with preds where genes do not appear in any
            # of the spots
            if gt.counts.shape[0] > preds.counts.shape[0]:
                _, gt_ind, _ = np.intersect1d(
                    gt.genes, preds.genes, return_indices=True
                )
                gt.counts = gt.counts[gt_ind]
                gt.genes = gt.genes[gt_ind]

            elif gt.counts.shape[0] < preds.counts.shape[0]:
                _, _, preds_ind = np.intersect1d(
                    gt.genes, preds.genes, return_indices=True
                )
                preds.counts = preds.counts[preds_ind]
                preds.genes = preds.genes[preds_ind]

            # due to padding, the shapes of the gene "images" may also not
            # be equal
            if gt.counts[0].shape != preds.counts[0].shape:
                min_r, min_c = np.minimum(
                    gt.counts[0].shape, preds.counts[0].shape
                )
                gt.counts = gt.counts[:, :min_r, :min_c]
                preds.counts = preds.counts[:, :min_r, :min_c]

            # normalize now that shapes are finalized
            gt.counts, preds.counts = normalize_with_shared_params(
                gt.counts, preds.counts
            )

            # crop mask to bbox, then resize to match counts (necessary
            # because of counts reshaping above)
            mask = mask[
                bbox.min_r : bbox.max_r + 1, bbox.min_c : bbox.max_c + 1
            ]
            mask = mask[: gt.counts.shape[1], : gt.counts.shape[2]]

            # checks
            logger.debug(
                f"gt shape: {gt.counts.shape}, pred shape: {preds.counts.shape}"
            )
            logger.debug(
                f"gt nans: {np.isnan(gt.counts).sum()}, pred nans: {np.isnan(preds.counts).sum()}"
            )
            logger.debug(
                f"gt counts size: {np.size(gt.counts)}, pred counts size: {np.size(preds.counts)}"
            )
            logger.debug(
                f"gt sum: {np.nansum(gt.counts)}, pred sum: {np.nansum(preds.counts)}"
            )
            logger.debug(
                f"gt range: [{np.nanmin(gt.counts)}, {np.nanmax(gt.counts)}], "
                f"pred range: [{np.nanmin(preds.counts)}, {np.nanmax(preds.counts)}]"
            )
            assert gt.counts.shape == preds.counts.shape
            assert np.all(np.strings.equal(gt.genes, preds.genes))

            # initialize arrays for retaining per-gene metrics
            n_genes = gt.genes.shape[0]
            rmses = np.zeros(n_genes)
            maes = np.zeros(n_genes)
            ssims = np.zeros(n_genes)

            logger.info(f"Calculating metrics for {d.name}.{sample_dir.name}")

            # Prepare arguments for parallel processing
            args_list = [
                (i, gt.counts[i], preds.counts[i], mask)
                for i in range(n_genes)
            ]

            # Use multiprocessing to calculate metrics in parallel
            with Pool(processes=int(cpu_count() * 0.75)) as pool:
                results = pool.map(calculate_gene_metrics, args_list)

            # Collect results
            for i, _rmse, _mae, _ssim in results:
                rmses[i] = _rmse
                maes[i] = _mae
                ssims[i] = _ssim

            metrics = {
                "genes": gt.genes,
                "rmse": rmses,
                "mae": maes,
                "ssim": ssims,
            }

            with open(sample_dir / "metrics.pickle", "wb") as f:
                pickle.dump(metrics, f)


def generate_plots():
    from src.analysis.postproc import plot_counts

    sample = "TENX147"

    base_dir = Path("/opt/gpudata/sjne/data_for_istar")
    dirs = [d for d in base_dir.iterdir() if d.stem.startswith("hex")]
    data_folder = "cnts-super"

    out_dir = base_dir / f"{sample}_CEACAM1_plots"
    out_dir.mkdir(exist_ok=True)

    for d in dirs:
        logger.info(f"Beginning processing {d.name}")

        for sample_dir in d.iterdir():
            if sample_dir.name != sample:
                continue

            logger.info(f"Processing {sample_dir.name}")

            with Image.open(sample_dir / "mask-small.png") as img:
                mask = np.array(img)

            # load ground truth
            mpp = d.stem.rsplit("_", maxsplit=1)[-1]
            sample_id = sample_dir.stem
            gt_base_dir = base_dir / f"sq_{mpp}" / sample_id
            gt_dir = gt_base_dir / data_folder

            bbox = get_gt_bounds(gt_base_dir / "cnts.tsv")
            # crop_and_save_he_image(sample_dir, bbox)

            gt = get_and_preprocess_data(data_dir=gt_dir, bbox=bbox)

            # load preds
            preds_dir = sample_dir / data_folder
            preds = get_and_preprocess_data(
                data_dir=preds_dir, mask=mask, bbox=bbox
            )

            # control for the case where the number of genes is not equal
            # typically occurs with preds where genes do not appear in any
            # of the spots
            if gt.counts.shape[0] > preds.counts.shape[0]:
                _, gt_ind, _ = np.intersect1d(
                    gt.genes, preds.genes, return_indices=True
                )
                gt.counts = gt.counts[gt_ind]
                gt.genes = gt.genes[gt_ind]

            elif gt.counts.shape[0] < preds.counts.shape[0]:
                _, _, preds_ind = np.intersect1d(
                    gt.genes, preds.genes, return_indices=True
                )
                preds.counts = preds.counts[preds_ind]
                preds.genes = preds.genes[preds_ind]

            # due to padding, the shapes of the gene "images" may also not
            # be equal
            if gt.counts[0].shape != preds.counts[0].shape:
                min_r, min_c = np.minimum(
                    gt.counts[0].shape, preds.counts[0].shape
                )
                gt.counts = gt.counts[:, :min_r, :min_c]
                preds.counts = preds.counts[:, :min_r, :min_c]

            # normalize now that shapes are finalized
            gt.counts, preds.counts = normalize_with_shared_params(
                gt.counts, preds.counts
            )

            # crop mask to bbox, then resize to match counts (necessary
            # because of counts reshaping above)
            mask = mask[
                bbox.min_r : bbox.max_r + 1, bbox.min_c : bbox.max_c + 1
            ]
            mask = mask[: gt.counts.shape[1], : gt.counts.shape[2]]

            # plot
            idx = np.where(gt.genes == "CEACAM1")[0][0]
            logger.debug(
                f"gt shape: {gt.counts.shape}, gt.ceacam shape: {gt.counts[idx].shape}, mask shape: {mask.shape}"
            )
            gt_plot = plot_counts(gt.counts[idx], mask)
            gt_plot.save(out_dir / f"{sample_dir.parent.name}-GT.png")
            gt_plot.close()
            pred_plot = plot_counts(preds.counts[idx], mask)
            pred_plot.save(out_dir / f"{sample_dir.parent.name}-INF.png")
            pred_plot.close()


if __name__ == "__main__":
    generate_plots()
    # from scipy.ndimage import uniform_filter, uniform_filter1d

    # # x = np.array([[26, 21, 5], [36, 28, 14], [34, 38, 43], [49, 46, 15]])
    # x = np.random.randint(0, 2, (5, 4)).astype(np.float32)
    # y = np.random.randint(0, 2, (5, 4)).astype(np.float32)

    # gt_val, gt_full = structural_similarity(
    #     x, y, win_size=3, full=True, data_range=1.0
    # )
    # print(gt_val)
    # print(gt_full)
    # print()
    # mine_val, mine_full = _ssim_nan(x, y, size=3, data_range=1.0)
    # print(mine_val)
    # print(mine_full)

    # # mask = np.random.randint(0, 2, (x.shape), dtype=bool)
    # # x[~mask] = np.nan
    # # y[~mask] = np.nan
    # # _ssim_nan(x, y)
