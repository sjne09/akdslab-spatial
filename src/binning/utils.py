import shapely


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
