import cudf
import shapely
from tifffile import RESUNIT, TiffFile

SCALE_FACTORS = {
    RESUNIT.INCH: 25.4e3,
    RESUNIT.CENTIMETER: 1.0e4,
    RESUNIT.MILLIMETER: 1.0e3,
    RESUNIT.MICROMETER: 1.0,
    RESUNIT.NONE: 1.0,
}


def approx_circle(a: float, b: float, r: float) -> shapely.Polygon:
    """
    Create an approximate circle as a Shapely Polygon.

    Parameters
    ----------
    a : float
        The x-coordinate of the circle's center

    b : float
        The y-coordinate of the circle's center

    r : float
        The radius of the circle

    Returns
    -------
    shapely.Polygon
        An approximate circle represented as a Shapely Polygon
    """
    return shapely.Point(a, b).buffer(r, quad_segs=64)


def get_mpp(tif: TiffFile, out_dir: str) -> float:
    """
    Get the microns per pixel (mpp) value from a TIFF file.

    Parameters
    ----------
    tif : TiffFile
        The TIFF file to extract the mpp from

    out_dir : str
        The directory to save the mpp value to

    Returns
    -------
    float
        The microns per pixel value
    """
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


def locs_from_tissue_positions(
    tissue_positions: cudf.DataFrame, out_dir: str
) -> None:
    """
    Converts a tissue positions matrix into the locs matrix expected
    by iStar.

    Parameters
    ----------
    tissue_positions : cudf.DataFrame
        The tissue positions matrix

    out_dir : str
        The directory to save the locs matrix to
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
