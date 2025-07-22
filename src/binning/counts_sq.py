import gc
import os
from pathlib import Path

import cudf
import cupy as cp
import pyarrow.dataset as ds


def determine_spots(transcript: cudf.DataFrame, spot_s: int) -> None:
    """
    Maps transcript coordinates to spot ids based on the side length
    of the square spots. Modifies `transcript` in place.

    Parameters
    ----------
    transcript : cudf.DataFrame
        A DataFrame containing the transcript data with columns
        "he_x", "he_y", and "feature_name"; "he_x" and "he_y" are in
        a coordinate space such that the coorinates can be divided by
        `spot_s` to determine the spot id

    spot_s : int
        The side length of the square spots, in pixels
    """
    coords = transcript[["he_x", "he_y"]].values
    spot_coords = cp.floor_divide(coords, spot_s).astype(cp.int32)
    transcript["spot_id"] = cudf.Series([f"{y}x{x}" for x, y in spot_coords])


def get_counts(
    transcript: cudf.DataFrame,
    spot_s: int,
    idx: str | int,
    out_dir: str,
) -> None:
    """
    Creates a gene expression count matrix, with rows as spots and
    columns as genes. Results are saved to a tsv

    Parameters
    ----------
    transcript : cudf.DataFrame
        A DataFrame containing the transcript data with columns
        "he_x", "he_y", and "feature_name"; "he_x" and "he_y" are in
        the same coordinate space as coords in `spots`

    spots : cudf.DataFrame
        A DataFrame containing the spot locations with columns
        "x", "y", and "spot"; "x" and "y" are in the same coordinate
        space as "he_x" and "he_y" in `transcript`

    spot_s : int
        The side length of the spots in pixels

    idx : str | int
        The index or unique identifier of the current batch of
        transcripts being processed

    out_dir : str
        The directory to save the output to
    """
    print(
        f"Process {idx} starting; ts size: {transcript.shape[0]}", flush=True
    )

    determine_spots(transcript, spot_s)
    print(f"Process {idx}: Spot ids determined. Binning", flush=True)

    bins = transcript[["spot_id", "feature_name"]].copy()
    bins["count"] = 1
    spot_counts_df: cudf.DataFrame = (
        bins.groupby(["spot_id", "feature_name"]).agg("sum").reset_index()
    )

    counts = spot_counts_df.to_pandas().pivot(
        # counts = spot_counts_df.pivot(
        index="spot_id",
        columns="feature_name",
        values="count",
    )
    counts.fillna(0, inplace=True)
    counts.to_csv(f"{out_dir}/cnts-{idx}.tsv", sep="\t")

    print(f"Process {idx} completed", flush=True)

    del transcript
    del counts


def join_cnts(dir_path: str, out_dir: str) -> None:
    """
    Joins gene count matrices into a single matrix. Saves to a tsv
    file.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing sub-matrices to be joined

    out_dir : str
        The directory to save the output file to
    """
    import pandas as pd  # cpu

    dir_path = Path(dir_path)
    iterdir = dir_path.iterdir()

    df = pd.read_csv(next(iterdir), sep="\t", index_col="spot_id")

    # initialize validation variables
    rolling_sum = df.sum(0).sum()
    rolling_cols = df.columns.to_series()
    rolling_idx = df.index.to_series()

    for f in iterdir:
        # read in matrix
        new = pd.read_csv(f, sep="\t", index_col="spot_id")

        # get stats for validation
        rolling_sum += new.sum(0).sum()
        rolling_cols = pd.concat([rolling_cols, new.columns.to_series()])
        rolling_idx = pd.concat([rolling_idx, new.index.to_series()])

        # join the new matrix with the existing matrix
        df = (
            pd.concat([df, new])
            .reset_index()
            .groupby("spot_id", sort=False)
            .agg("sum")
        )

    # ensure the combined data matches the content of the component files
    print(rolling_sum == df.sum(0).sum())
    print(set(rolling_cols.to_list()).difference(set(df.columns.to_list())))
    print(set(rolling_idx.to_list()).difference(set(df.index.to_list())))

    # save results
    df.to_csv(f"{out_dir}/cnts.tsv", sep="\t")


def process_sample(sample_id: str, output_dir: str) -> None:
    """
    Processes a single sample's transcript data to generate gene
    expression counts within spots.

    Parameters
    ----------
    sample_id : str
        The unique identifier for the sample, used to locate the
        transcript data

    output_dir : str
        The directory containing input data and to save the output
        files to. Must contain a file named "side-raw.txt" with the
        side length of the square spots in pixels
    """
    print(f"Processing {sample_id}...")

    with open(f"{output_dir}/side-raw.txt", "r") as f:
        spot_s = float(f.read())

    pf = ds.dataset(
        f"/opt/gpudata/sjne/HEST/data/transcripts/{sample_id}_transcripts.parquet"
    )
    os.makedirs(f"{output_dir}/cnts", exist_ok=True)

    batch_idx = 0
    for batch in pf.to_batches(batch_size=500000):
        pandas_df = batch.to_pandas()

        for col in pandas_df.columns:
            if pandas_df[col].dtype == "object":
                pandas_df[col] = pandas_df[col].apply(
                    lambda x: x.decode("utf-8", errors="replace")
                    if isinstance(x, bytes)
                    else x
                )

        ts = cudf.from_pandas(pandas_df)

        get_counts(ts, spot_s, batch_idx, f"{output_dir}/cnts")
        batch_idx += 1

        del ts, pandas_df
        gc.collect()
