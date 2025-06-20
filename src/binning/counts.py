import pandas as pd
import numpy as np
from shapely import Point
from scipy.spatial.distance import cdist

from src.binning.utils import approx_circle

from pandas._libs.missing import NAType


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
    transcript: pd.DataFrame, spots: pd.DataFrame, spot_r: float
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
    counts = spot_counts_df.pivot(
        index="spot_id",
        columns="feature_name",
        values="count",
    )
    counts[counts.isna()] = 0
    print(counts.shape)
    counts.to_csv("cnts.tsv", sep="\t")

    del transcript
    del counts


if __name__ == "__main__":
    locs = pd.read_csv("locs-raw.tsv", sep="\t")
    spot_r = 100.43759671802853
    x = np.random.randint(int(locs["x"].min()), int(locs["x"].max()))
    y = np.random.randint(int(locs["y"].min()), int(locs["y"].max()))
    # print(determine_spot(x, y, locs, spot_r))
    start = np.random.randint(0, 20000)
    ts = pd.read_parquet(
        "/opt/gpudata/sjne/HEST/data/transcripts/TENX111_transcripts.parquet"
    )

    get_counts(ts, locs, spot_r)
