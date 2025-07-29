import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image


def get_counts_matrix(
    counts: pd.Series, mask: np.ndarray, normalize: bool
) -> np.ndarray:
    """
    Creates a matrix of gene expression counts for a single gene.
    The matrix is generated with a shape matching the input mask.
    Masked out regions will have a value of NaN.

    Parameters
    ----------
    counts : pd.Series
        Expression data for a single gene across all spots. Index must
        be a spot id of the form "RxC", where R is the array_row and C
        is the array_col from the tissue positions file (standard
        output from Visium)

    mask : np.ndarray
        A mask for the original HE image defining the tissue limits.
        The shape of the mask must correspond to the spot id
        coordinates from `counts`, i.e., the resolution of the mask
        must match the resolution of the gene expression data

    normalize : bool
        Whether to apply min-max normalization to the counts data

    Returns
    -------
    np.ndarray
        The counts matrix
    """
    counts_array = np.zeros(mask.shape)
    for spot_id in counts.index:
        # unpack the index to get coords, which are 'RxC'
        spot_row, spot_col = map(lambda x: int(x), spot_id.split("x"))

        # ignore index OOB
        try:
            counts_array[spot_row, spot_col] = counts[spot_id]
        except IndexError:
            continue

    counts_array[~mask] = np.nan

    # min-max normalize
    if normalize:
        counts_array -= np.nanmin(counts_array)
        counts_array /= np.nanmax(counts_array) + 1e-12

    return counts_array


def plot_counts(counts: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Visualizes the counts matrix as an image.

    Parameters
    ----------
    counts : np.ndarray
        The counts matrix, with NaN values for masked out regions

    mask : np.ndarray
        A mask for the original HE image defining the tissue limits.
        The shape of the mask must correspond to the resolution of the
        counts data

    Returns
    -------
    Image
        A PIL Image object representing the counts matrix
    """
    cmap = plt.get_cmap("turbo")

    # RGB
    heatmap = cmap(counts)[..., :3]

    # set masked out areas to white
    heatmap[~mask] = 1.0

    # convert from float to int and save
    heatmap = (heatmap * 255).astype(np.uint8)
    return Image.fromarray(heatmap)


def process_sample(
    counts: pd.DataFrame,
    mask: np.ndarray,
    plot_outdir: Path,
    vector_outdir: Path,
) -> None:
    """
    Processes a single sample's gene expression data, generating
    visualizations and saving the counts data in a structured format.

    Parameters
    ----------
    counts : pd.DataFrame
        A DataFrame containing gene expression counts with spot ids as
        the index. The index should be of the form "RxC", where R is
        the array_row and C is the array_col from the tissue positions
        file (standard output from Visium)

    mask : np.ndarray
        A mask for the original HE image defining the tissue limits.
        The shape of the mask must correspond to the resolution of the
        gene expression data

    plot_outdir : Path
        The directory to save the visualizations to

    vector_outdir : Path
        The directory to save the counts matrices to
    """
    for gene in counts.columns:
        gene_counts = get_counts_matrix(
            counts=counts[gene], mask=mask, normalize=False
        )
        plot = plot_counts(counts=gene_counts, mask=mask)

        with open(vector_outdir / f"{gene}.pickle", "wb") as f:
            pickle.dump(gene_counts, f)

        plot.save(plot_outdir / f"{gene}.png")
        plot.close()


def process_single_sample(sample_path: Path) -> None:
    """
    Process a single sample - designed to be called by multiprocessing.

    Parameters
    ----------
    sample_path : Path
        Path to the sample directory
    """
    logger.info(f"Processing {sample_path.parent}.{sample_path.name}")

    cnts_super_dir = Path(sample_path / "cnts-super")
    cnts_super_dir.mkdir(exist_ok=True)

    cnts_super_plots_dir = Path(sample_path / "cnts-super-plots")
    cnts_super_plots_dir.mkdir(exist_ok=True)

    counts_df = pd.read_csv(
        sample_path / "cnts.tsv", sep="\t", index_col="spot_id"
    )

    with Image.open(sample_path / "mask-small.png") as img:
        mask = np.array(img)

    process_sample(
        counts=counts_df,
        mask=mask,
        plot_outdir=cnts_super_plots_dir,
        vector_outdir=cnts_super_dir,
    )
    logger.info(f"Done processing {sample_path.parent}.{sample_path.name}")


if __name__ == "__main__":
    base_dir = Path("/opt/gpudata/sjne/data_for_istar")
    dirs = [d for d in base_dir.iterdir() if d.name.startswith("sq")]

    # Collect all sample paths
    all_samples = []
    for d in dirs:
        for sample in d.iterdir():
            all_samples.append(sample)

    # use multiprocessing to process samples in parallel
    num_processes = max(1, int(cpu_count() * 0.50))

    logger.info(
        f"Processing {len(all_samples)} samples using {num_processes} processes"
    )

    with Pool(processes=num_processes) as pool:
        pool.map(process_single_sample, all_samples)

    logger.info("Done processing all samples")


# if __name__ == "__main__":
#     base_dir = Path("/opt/gpudata/sjne/data_for_istar")
#     dirs = [d for d in base_dir.iterdir() if d.name.startswith("sq")]

#     for d in dirs:
#         print(f"Beginning processing {d.name}")

#         for sample in d.iterdir():
#             print(f"Processing {sample.name}")

#             cnts_super_dir = Path(sample / "cnts-super")
#             cnts_super_dir.mkdir(exist_ok=True)

#             cnts_super_plots_dir = Path(sample / "cnts-super-plots")
#             cnts_super_plots_dir.mkdir(exist_ok=True)

#             counts_df = pd.read_csv(
#                 sample / "cnts.tsv", sep="\t", index_col="spot_id"
#             )

#             with Image.open(sample / "mask-small.png") as img:
#                 mask = np.array(img)

#             process_sample(
#                 counts=counts_df,
#                 mask=mask,
#                 plot_outdir=cnts_super_plots_dir,
#                 vector_outdir=cnts_super_dir,
#             )

#         print(f"Done processing {d.name}\n")
