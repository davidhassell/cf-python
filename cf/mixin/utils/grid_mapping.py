"""Utilities for creating `pyproj.CRS` instances."""

import logging

from cfdm import is_log_level_info

logger = logging.getLogger(__name__)


def get_ellipsoid_parameters(cr):
    """Get ellipsoid parmaeters from a coordinate reference construct.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference` or `None`
            The coordinate reference construct, or `None`, in which
            case the CF defualt ellpsoid is assumed.

    :Returns:

        `dict`
            The `pyproj.CRS` ellpsoid parameters.

    """
    kwargs = {}
    if cr is None:
        p = {}
    else:
        p = cr.coordinate_conversion.parameters()
        if "reference_ellipsoid_name" in p:
            kwargs["ellps"] = p["reference_ellipsoid_name"]

        if "semi_major_axis" in p:
            kwargs["a"] = p["semi_major_axis"]

        if "semi_minor_axis" in p:
            kwargs["b"] = p["semi_minor_axis"]

        if "inverse_flattening" in p:
            kwargs["rf"] = p["inverse_flattening"]

    if not kwargs:
        kwargs = {"ellps": "sphere"}

    kwargs["R"] = p.get("earth_radius")

    prime_meridian_name = p.get("prime_meridian_name")
    if prime_meridian_name is not None:
        kwargs["pm"] = prime_meridian_name
    else:
        kwargs["pm"] = p.get("longitude_of_prime_meridian", 0)

    return kwargs


def create_proj_CRS(kwargs, cr):
    """Create a `pyproj.CRS` instance.

    .. versionadded:: NEXTVERSION

    :Parameters:

        kwargs: `dict`
            A dictionary of keyword arguments for initialising the the
            `pyproj.CRS` instance.

        cr: `CoordinateReference`
            The coordinate reference construct from which *kwargs* was
            derived.

    :Returns:

        `pyproj.CRS` or `None`
            The created CRS, or `None` if one couldn't be created.

    """
    import pyproj

    # Create the `pyproj.CRS` keywword arguments, which include
    # parameters for describing the ellipsoid
    kwargs = get_ellipsoid_parameters(cr) | kwargs

    # Remove `None` values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        proj = pyproj.CRS(**kwargs)
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d latitude and longitude coordinates "
                f"for {cr!r}: Bad pyproj.CRS parameters: {kwargs!r}"
            )  # pragma: no cover

        return

    return proj


# ====================================================================
# Functions for creating `pyproj.CRS` instances for each CF grid
# mapping type.
#
# These functions are called by `_create_2d_latlon_coordinates`.
# ====================================================================


def albers_equal_area(cr):
    """Create an azimuthal_equidistant CRS.

    https://proj.org/en/stable/operations/projections/aea.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "aea",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_central_meridian"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    lat_2 = None
    standard_parallel = p.get("standard_parallel")
    try:
        lat_1 = standard_parallel[0]
    except Exception:
        lat_1 = standard_parallel
    else:
        try:
            lat_2 = standard_parallel[1]
        except Exception:
            pass

    kwargs["lat_1"] = lat_1
    kwargs["lat_2"] = lat_2

    return create_proj_CRS(kwargs, cr)


def azimuthal_equidistant(cr):
    """Create an azimuthal_equidistant CRS.

    https://proj.org/en/stable/operations/projections/aeqd.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "aeqd",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    return create_proj_CRS(kwargs, cr)


def geostationary(cr):
    """Create a geostationary CRS.

    https://proj.org/en/stable/operations/projections/geos.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "geos",
        "h": p.get("perspective_point_height"),
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    sweep_angle_axis = p.get("sweep_angle_axis")
    fixed_angle_axis = p.get("fixed_angle_axis")
    match sweep_angle_axis:
        case "x":
            ok = fixed_angle_axis in (None, "y")
        case "y":
            ok = fixed_angle_axis in (None, "x")
        case None:
            ok = True
            if fixed_angle_axis == "x":
                sweep_angle_axis = "y"
            elif fixed_angle_axis == "y":
                sweep_angle_axis = "x"
            else:
                ok = False
        case _:
            ok = False

    if not ok:
        logger.info(
            "Can't create 2-d latitude and longitude coordinates "
            f"for {cr!r}: Bad 'sweep_angle_axis' parameter: "
            f"{sweep_angle_axis!r}, or bad 'fixed_angle_axis' "
            f"parameter: {fixed_angle_axis!r}"
        )  # pragma: no cover

    kwargs["sweep"] = sweep_angle_axis

    return create_proj_CRS(kwargs, cr)


def lambert_azimuthal_equal_area(cr):
    """Create a lambert_azimuthal_equal_area CRS.

    https://proj.org/en/stable/operations/projections/laea.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "laea",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }
    return create_proj_CRS(kwargs, cr)


def lambert_conformal_conic(cr):
    """Create a lambert_conformal_conic CRS.

    https://proj.org/en/stable/operations/projections/lcc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "lcc",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    lat_2 = None
    standard_parallel = p.get("standard_parallel")
    try:
        lat_1 = standard_parallel[0]
    except Exception:
        lat_1 = standard_parallel
    else:
        try:
            lat_2 = standard_parallel[1]
        except Exception:
            pass

    kwargs["lat_1"] = lat_1
    kwargs["lat_2"] = lat_2

    return create_proj_CRS(kwargs, cr)


def lambert_cylindrical_equal_area(cr):
    """Create a lambert_cylindrical_equal_area CRS.

    https://proj.org/en/stable/operations/projections/cea.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "cea",
        "lon_0": p.get("longitude_of_central_meridian"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    standard_parallel = p.get("standard_parallel")
    if standard_parallel is not None:
        kwargs["lat_ts"] = standard_parallel
    else:
        kwargs["k_0"] = p.get("scale_factor_at_projection_origin")

    return create_proj_CRS(kwargs, cr)


def latitude_longitude(cr):
    """create a latitude_longitude CRS.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The latitude_longitude coordinate reference construct from
            which to create the CRS, or `None` if there isn't one (in
            which case a spherical CRS is created).

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    kwargs = {"proj": "longlat"}
    return create_proj_CRS(kwargs, cr)


def mercator(cr):
    """Create a mercator CRS.

    https://proj.org/en/stable/operations/projections/merc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "merc",
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    standard_parallel = p.get("standard_parallel")
    if standard_parallel is not None:
        kwargs["lat_ts"] = standard_parallel
    else:
        kwargs["k_0"] = p.get("scale_factor_at_projection_origin")

    return create_proj_CRS(kwargs, cr)


def oblique_mercator(cr):
    """Create an oblique_mercator CRS.

    https://proj.org/en/stable/operations/projections/omerc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "omerc",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "alpha": p.get("azimuth_of_central_line"),
        "k_0": p.get("scale_factor_at_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }
    return create_proj_CRS(kwargs, cr)


def orthographic(cr):
    """Create an orthographic CRS.

    https://proj.org/en/stable/operations/projections/ortho.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "ortho",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }
    return create_proj_CRS(kwargs, cr)


def polar_stereographic(cr):
    """Create a polar_stereographic CRS.

    https://proj.org/en/stable/operations/projections/stere.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "stere",
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    longitude_of_projection_origin = p.get("longitude_of_projection_origin")
    if longitude_of_projection_origin is not None:
        kwargs["lon_0"] = longitude_of_projection_origin
    else:
        kwargs["lon_0"] = p.get("straight_vertical_longitude_from_pole")

    standard_parallel = p.get("standard_parallel")
    if standard_parallel is not None:
        kwargs["lat_ts"] = standard_parallel
    else:
        kwargs["k_0"] = p.get("scale_factor_at_projection_origin")

    latitude_of_projection_origin = p.get("latitude_of_projection_origin")
    try:
        ok = (
            latitude_of_projection_origin == -90
            or latitude_of_projection_origin == 90
        )
    except Exception:
        ok = False

    if not ok:
        logger.info(
            "Can't create 2-d latitude and longitude coordinates "
            f"for {cr!r}: Bad 'latitude_of_projection_origin' parameter: "
            f"{latitude_of_projection_origin!r}"
        )  # pragma: no cover

    kwargs["lat_0"] = latitude_of_projection_origin

    return create_proj_CRS(kwargs, cr)


def rotated_latitude_longitude(cr):
    """Create a rotated_latitude_longitude CRS`.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "ob_tran",
        "o_proj": "longlat",
        "o_lon_p": p.get("north_pole_grid_longitude", 0),
        "o_lat_p": p.get("grid_north_pole_latitude"),
    }

    grid_north_pole_longitude = p.get("grid_north_pole_longitude")
    try:
        kwargs["lon_0"] = float(grid_north_pole_longitude) + 180
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d latitude and longitude coordinates "
                f"for {cr!r}: Bad 'grid_north_pole_longitude' parameter: "
                f"{grid_north_pole_longitude!r}"
            )  # pragma: no cover

        return

    return create_proj_CRS(kwargs, cr)


def sinusoidal(cr):
    """Create a sinusoidal CRS.

    https://proj.org/en/stable/operations/projections/sinu.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "sinu",
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    return create_proj_CRS(kwargs, cr)


def stereographic(cr):
    """Create a stereographic CRS.

    https://proj.org/en/stable/operations/projections/stere.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "stere",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "k_0": p.get("scale_factor_at_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }
    return create_proj_CRS(kwargs, cr)


def transverse_mercator(cr):
    """Create a tranverse_mercator CRS.

    https://proj.org/en/stable/operations/projections/tmerc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "tmerc",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_central_meridian"),
        "k_0": p.get("scale_factor_at_central_meridian"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }

    return create_proj_CRS(kwargs, cr)


def vertical_perspective(cr):
    """Create a vertical_perspective CRS.

    https://proj.org/en/stable/operations/projections/nsper.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "nsper",
        "h": p.get("perspective_point_height"),
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting", 0),
        "y_0": p.get("false_northing", 0),
    }
    return create_proj_CRS(kwargs, cr)
