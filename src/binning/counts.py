import argparse
import os
from pathlib import Path
from multiprocessing import Pool, Queue, current_process

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs.missing import NAType
from pyarrow.parquet import ParquetFile
from scipy.spatial.distance import cdist
from shapely import Point

from src.binning.utils import approx_circle


def determine_spot(
    x: float, y: float, spots: pd.DataFrame, spot_r: float
) -> str | NAType:
    """
    Determine the spot ID that a given coordinate (x, y) is contained
    within.

    Parameters
    ----------
    x : float
        The x-coordinate of the point to check

    y : float
        The y-coordinate of the point to check

    spots : pd.DataFrame
        A DataFrame containing the spot locations with columns
        "x", "y", and "spot"; "x" and "y" are in the same coordinate
        space as the coordinates being checked

    spot_r : float
        The radius of the spots in pixels

    Returns
    -------
    str | NAType
        The spot ID if the point is contained within a spot, otherwise
        returns pd.NA
    """
    # FIXME maybe use geopandas and cuspatial for this?
    distances = cdist(
        np.array([[x, y]]), spots[["x", "y"]].values, metric="euclidean"
    ).squeeze()
    nearest_spot = spots.iloc[distances.argmin()]

    circle = approx_circle(*nearest_spot[["x", "y"]].values, spot_r)

    if circle.contains(Point(x, y)):
        return nearest_spot["spot"]

    return pd.NA


def get_counts(
    transcript: pd.DataFrame,
    spots: pd.DataFrame,
    spot_r: float,
    idx: int,
    out_dir: str,
) -> None:
    """
    Get a gene expression count matrix, with rows as spots and columns
    as genes.

    Parameters
    ----------
    transcript : pd.DataFrame
        A DataFrame containing the transcript data with columns
        "he_x", "he_y", and "feature_name"; "he_x" and "he_y" are in
        the same coordinate space as coords in `spots`

    spots : pd.DataFrame
        A DataFrame containing the spot locations with columns
        "x", "y", and "spot"; "x" and "y" are in the same coordinate
        space as "he_x" and "he_y" in `transcript`

    spot_r : float
        The radius of the spots in pixels
    """
    print(
        f"Process {idx} starting; ts size: {transcript.shape[0]}", flush=True
    )

    # get the spot id for each entry in the transcript matrix
    transcript["spot_id"] = transcript.apply(
        lambda row: determine_spot(row["he_x"], row["he_y"], spots, spot_r),
        axis=1,
    )

    bins = transcript[["spot_id", "feature_name"]].copy()
    bins["count"] = 1
    spot_counts_df: pd.DataFrame = (
        bins.groupby(["spot_id", "feature_name"]).agg("sum").reset_index()
    )

    # spot_counts_df["count"] = spot_counts_df["count"].astype(int)
    counts = spot_counts_df.pivot(
        index="spot_id",
        columns="feature_name",
        values="count",
    )
    counts[counts.isna()] = 0
    counts.to_csv(f"{out_dir}/cnts-{idx}.tsv", sep="\t")

    print(f"Process {idx} completed", flush=True)

    del transcript
    del counts


def worker(
    q: Queue, pf: ParquetFile, spots: pd.DataFrame, spot_r: float, out_dir: str
) -> None:
    worker_id = current_process()._identity[0]
    print(f"Worker {worker_id} starting", flush=True)

    idx = 0
    while True:
        batch = q.get()
        if batch is None:
            break

        ts = pf.read_row_group(batch).to_pandas()
        get_counts(ts, spots, spot_r, f"{worker_id}:{idx}", out_dir)

        idx += 1

    print(f"Worker {worker_id} completed", flush=True)


def join_cnts(dir_path: str, out_dir: str) -> None:
    dir_path = Path(dir_path)
    iterdir = dir_path.iterdir()

    df = pd.read_csv(next(iterdir), sep="\t", index_col="spot_id")
    rolling_sum = df.sum(0).sum()
    rolling_cols = df.columns.to_series()
    rolling_idx = df.index.to_series()

    for f in iterdir:
        new = pd.read_csv(f, sep="\t", index_col="spot_id")
        rolling_sum += new.sum(0).sum()
        rolling_cols = pd.concat([rolling_cols, new.columns.to_series()])
        rolling_idx = pd.concat([rolling_idx, new.index.to_series()])

        df = (
            pd.concat([df, new])
            .reset_index()
            .groupby("spot_id", sort=False)
            .agg("sum")
        )

    df.to_csv(f"{out_dir}/cnts.tsv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workers")
    args = parser.parse_args()

    sample_ids = ["TENX111", "TENX114", "TENX147", "TENX148", "TENX149"]
    excl = {"TENX111"}

    for sample_id in sample_ids:
        locs = pd.read_csv(f"{sample_id}/locs-raw.tsv", sep="\t")

        with open(f"{sample_id}/radius-raw.txt", "r") as f:
            spot_r = float(f.read())

        pf = ParquetFile(
            f"/opt/gpudata/sjne/HEST/data/transcripts/{sample_id}_transcripts.parquet"
        )
        os.makedirs(f"{sample_id}/cnts", exist_ok=True)

        workers = int(args.workers)
        q = Queue(maxsize=workers)
        pool = Pool(
            workers,
            initializer=worker,
            initargs=(q, pf, locs, spot_r, f"{sample_id}/cnts"),
        )

        for i in range(pf.num_row_groups):
            q.put(i)

        for i in range(workers):
            q.put(None)

        pool.close()
        pool.join()

        join_cnts(f"{sample_id}/cnts", sample_id)
