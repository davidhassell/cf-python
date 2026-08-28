"""Utilities for creating `pyproj.CRS` instances.

:Glossary:

Definitions of `pyproj.CRS` parameters that map to CF grid mapping
parameters.

For further details see
https://proj.org/en/stable/operations/projections and
https://github.com/cf-convention/cf-conventions/wiki/Mapping-from-CF-Grid-Mapping-Attributes-to-CRS-WKT-Elements

* a: Semi-major axis of the ellipsoid.

* alpha: Azimuth of centerline clockwise from north at the centre
         point of the line. If gamma is not given then alpha
         determines the value of gamma.

* b: Semi-minor axis of the ellipsoid.

* ellps: The name of a built-in ellipsoid definition.

* f: Flattening of the ellipsoid.

* h: Height of the view point above the Earth and must be in the same
     units as the radius of the sphere or semimajor axis of the
     ellipsoid.

* k_0: Scale factor. Determines scale factor used in the projection.

* lat_0: Latitude of natural origin, latitude of false origin or
         latitude of projection centre (naming and meaning depend on
         the projection method).

* lat_1: First standard parallel.

* lat_2: Second standard parallel.

* lat_ts: Defines the latitude where scale is not distorted. It is
          only taken into account for Polar Stereographic formulations
          (lat_0 = +/- 90 ), and then defaults to the lat_0 value. If
          set to a value different from +/- 90, it takes precedence
          over k_0 if both options are used together.

* lon_0: Central meridian/longitude of natural origin, longitude of
         origin or longitude of false origin (naming and meaning
         depend on the projection method).

* o_lat_p: Latitude of the North pole of the unrotated source CRS,
           expressed in the rotated geographic CRS.

* o_lon_p: Longitude of the North pole of the unrotated source CRS,
           expressed in the rotated geographic CRS.

* o_proj: Oblique projection.

* pm: Prime meridian.

* R: Radius of the sphere, given in meters. If used in conjunction
     with ellps, R takes precedence.

* rf: Reverse flattening of the ellipsoid, 1/f

* sweep: Sweep angle axis of the viewing instrument. Valid options are
         "x" and "y".

* y_0: False northing, northing at false origin or northing at
       projection centre (naming and meaning depend on the projection
       method). Always in meters.

* x_0: False easting, easting at false origin or easting at projection
       centre (naming and meaning depend on the projection
       method). Always in meters.

"""

import logging
import warnings

from cfdm import is_log_level_debug, is_log_level_info

# Suppress warning about lossy WKT-to-PROJ conversion,it only refers
# to lost information that doesn't affect the transformation.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*lose important projection information.*",
)

logger = logging.getLogger(__name__)


def _ellipsoid_parameters(cr):
    """Get ellipsoid parameters from a coordinate reference construct.

    https://proj.org/en/stable/usage/ellipsoids.html

    https://proj.org/en/stable/usage/projections.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct, or `None`, in which
            case the CF default ellipsoid is assumed.

    :Returns:

        `dict`
            The `pyproj.CRS` ellipsoid parameters.

    """
    kwargs = {}

    p = cr.datum.parameters()
    crs_wkt = cr.coordinate_conversion.get_parameter("crs_wkt", None)

    inverse_flattening = p.get("inverse_flattening")
    semi_major_axis = p.get("semi_major_axis")
    semi_minor_axis = p.get("semi_minor_axis")
    earth_radius = p.get("earth_radius")
    reference_ellipsoid_name = p.get("reference_ellipsoid_name")

    if inverse_flattening == 0:
        # Sphere
        if semi_major_axis is not None:
            kwargs["R"] = semi_major_axis
        elif earth_radius is not None:
            kwargs["R"] = earth_radius
        elif not crs_wkt and reference_ellipsoid_name is None:
            reference_ellipsoid_name = "sphere"
    else:
        # Ellipsoid
        if earth_radius is not None:
            kwargs["R"] = earth_radius
        else:
            if semi_major_axis is not None:
                kwargs["a"] = semi_major_axis

            if semi_minor_axis is not None:
                kwargs["b"] = semi_minor_axis

            if inverse_flattening is not None:
                kwargs["rf"] = inverse_flattening

    if reference_ellipsoid_name is not None:
        kwargs["ellps"] = reference_ellipsoid_name

    if not crs_wkt and not kwargs:
        # Default to a sphere, in the absence of other information.
        kwargs = {"ellps": "sphere"}

    prime_meridian_name = p.get("prime_meridian_name")
    if prime_meridian_name is not None:
        kwargs["pm"] = prime_meridian_name
    elif not crs_wkt:
        kwargs["pm"] = p.get("longitude_of_prime_meridian", 0)

    return kwargs


def _crs_wkt_parameters(cr):
    """Get `pyproj.CRS` parameters from a crs_wkt parameter.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.

    :Returns:

        `dict`
            The `pyproj.CRS` parameters derived from coordinate
            reference construct crs_wkt parameters, if any.

    """

    crs_wkt = cr.coordinate_conversion.get_parameter("crs_wkt", None)
    if crs_wkt is None:
        return {}

    import pyproj

    return pyproj.CRS.from_wkt(crs_wkt).to_dict()


def _create_pyproj_CRS(kwargs, cr, ellipsoid_only=False):
    """Create a `pyproj.CRS` instance.

    .. versionadded:: NEXTVERSION

    :Parameters:

        kwargs: `dict`
            A dictionary of keyword arguments for initialising the
            `pyproj.CRS` instance.

            The keyword arguments should not include a description of
            the ellipsoid, as this is automatically derived from *cr*.

            If the *cr* contains a ``crs_wkt`` parameter then it is
            converted to `pyproj.CRS` keyword arguments that are
            automatically included.

        cr: `CoordinateReference`
            The coordinate reference construct from which *kwargs* was
            derived.

        ellipsoid_only: `bool`, optional
            Return the CRS defined only by the ellipsoid and prime
            meridian.

    :Returns:

        `pyproj.CRS` or `None`
            The created CRS, or `None` if one couldn't be created.

    """
    import pyproj

    # Remove `None` values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Specify the units of the 1-d coordinates
    kwargs["units"] = "m"

    kwargs = _crs_wkt_parameters(cr) | _ellipsoid_parameters(cr) | kwargs

    try:
        proj = pyproj.CRS(**kwargs)
    except Exception as error:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create a pyproj.CRS for {cr!r}: {error}"
            )  # pragma: no cover

        return

    if (
        ellipsoid_only
        and cr.coordinate_conversion.get_parameter("grid_mapping_name", None)
        != "latitude_longitude"
    ):
        # Return the CRS defined by the ellipsoid and prime meridian
        # of a non-latitude_longitude coordinate reference
        proj = proj.geodetic_crs

    if is_log_level_debug(logger):
        logger.debug(f"pyproj.CRS: {proj}")

    return proj


def _cc_parameter(p, parameter, default=None):
    """Get a parameter from a coordinate reference construct.

    If there is a ``crs_wkt`` parameter then:

    * `None` will be returned if the *parameter* does not exist.

    If there is not a ``crs_wkt`` parameter then:

    * If *default* is not `None`, then that *default* will be returned
      if the *parameter* does not exist.

    * If *default* is `None`, then a `KeyError` will be raised if the
      *parameter* does not exist.

    This behaviour allows a ``crs_wkt`` parameter to provide a value
    for a missing CF grid mapping parameter (which happens later on in
    `_create_pyproj_CRS`).

    .. versionadded:: NEXTVERSION

    :Parameters:

        p: `dict`
            A dictionary of the coordinate reference construct
            parameters.

        parameter: `str`
            The name of the parameter.

        default: optional
            What to do if the parameter doesn't not exist (see above).

    :Returns:

            The parameter value.

    """
    if "crs_wkt" in p:
        return p.get(parameter)

    if default is not None:
        return p.get(parameter, default)

    return p[parameter]


# ====================================================================
# Functions for creating a `pyproj.CRS` instance for each CF grid
# mapping type.
# ====================================================================


def albers_equal_area(cr):
    """Create an albers_equal_area CRS.

    https://proj.org/en/stable/operations/projections/aea.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from which the CRS is
            derived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "aea",
        "lon_0": _cc_parameter(p, "longitude_of_central_meridian"),
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    standard_parallel = _cc_parameter(p, "standard_parallel")
    if standard_parallel is not None:
        try:
            lat_1 = standard_parallel[0]
        except Exception:
            lat_1 = standard_parallel
        else:
            try:
                kwargs["lat_2"] = standard_parallel[1]
            except Exception:
                pass

        kwargs["lat_1"] = lat_1

    return _create_pyproj_CRS(kwargs, cr)


def azimuthal_equidistant(cr):
    """Create an azimuthal_equidistant CRS.

    https://proj.org/en/stable/operations/projections/aeqd.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "aeqd",
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    return _create_pyproj_CRS(kwargs, cr)


def geostationary(cr):
    """Create a geostationary CRS.

    https://proj.org/en/stable/operations/projections/geos.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    crs_wkt = "crs_wkt" in p

    kwargs = {
        "proj": "geos",
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "h": _cc_parameter(p, "perspective_point_height"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
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

    if not crs_wkt and not ok:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create coordinates for {cr!r}: "
                f"Bad 'sweep_angle_axis' parameter: {sweep_angle_axis!r}, "
                f"or bad 'fixed_angle_axis' parameter: {fixed_angle_axis!r}"
            )  # pragma: no cover

        return

    kwargs["sweep"] = sweep_angle_axis

    if p.get("latitude_of_projection_origin", 0) != 0:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create coordinates for {cr!r}: "
                "Bad 'latitude_of_projection_origin' parameter: "
                f"{p['latitude_of_projection_origin']!r}"
            )  # pragma: no cover

        return

    return _create_pyproj_CRS(kwargs, cr)


def lambert_azimuthal_equal_area(cr):
    """Create a lambert_azimuthal_equal_area CRS.

    https://proj.org/en/stable/operations/projections/laea.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "laea",
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }
    return _create_pyproj_CRS(kwargs, cr)


def lambert_conformal_conic(cr):
    """Create a lambert_conformal_conic CRS.

    https://proj.org/en/stable/operations/projections/lcc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "lcc",
        "lon_0": _cc_parameter(p, "longitude_of_central_meridian"),
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    standard_parallel = _cc_parameter(p, "standard_parallel")
    if standard_parallel is not None:
        try:
            lat_1 = standard_parallel[0]
        except Exception:
            lat_1 = standard_parallel
        else:
            try:
                kwargs["lat_2"] = standard_parallel[1]
            except Exception:
                pass

        kwargs["lat_1"] = lat_1

    return _create_pyproj_CRS(kwargs, cr)


def lambert_cylindrical_equal_area(cr):
    """Create a lambert_cylindrical_equal_area CRS.

    https://proj.org/en/stable/operations/projections/cea.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    crs_wkt = "crs_wkt" in p

    kwargs = {
        "proj": "cea",
        "lon_0": _cc_parameter(p, "longitude_of_central_meridian"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    standard_parallel = _cc_parameter(p, "standard_parallel")
    if standard_parallel is not None:
        kwargs["lat_ts"] = standard_parallel
    elif not crs_wkt:
        kwargs["k_0"] = p["scale_factor_at_projection_origin"]

    return _create_pyproj_CRS(kwargs, cr)


def latitude_longitude(cr):
    """Create a latitude_longitude CRS.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from which to create
            the CRS, or `None` if there isn't one (in which case a
            spherical CRS is created).

            .. note:: Only the datum parameters of *cr* are used, so
                      the coordinate reference construct does not need
                      to be a latitude_longitude grid mapping.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    kwargs = {"proj": "longlat"}

    return _create_pyproj_CRS(kwargs, cr, ellipsoid_only=True)


def mercator(cr):
    """Create a mercator CRS.

    https://proj.org/en/stable/operations/projections/merc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    crs_wkt = "crs_wkt" in p

    kwargs = {
        "proj": "merc",
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    standard_parallel = _cc_parameter(p, "standard_parallel")
    if standard_parallel is not None:
        kwargs["lat_ts"] = standard_parallel
    elif not crs_wkt:
        kwargs["k_0"] = p["scale_factor_at_projection_origin"]

    return _create_pyproj_CRS(kwargs, cr)


def oblique_mercator(cr):
    """Create an oblique_mercator CRS.

    https://proj.org/en/stable/operations/projections/omerc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "omerc",
        "alpha": _cc_parameter(p, "azimuth_of_central_line"),
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "lonc": _cc_parameter(p, "longitude_of_projection_origin"),
        "k_0": _cc_parameter(p, "scale_factor_at_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }
    return _create_pyproj_CRS(kwargs, cr)


def orthographic(cr):
    """Create an orthographic CRS.

    https://proj.org/en/stable/operations/projections/ortho.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "ortho",
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }
    return _create_pyproj_CRS(kwargs, cr)


def polar_stereographic(cr):
    """Create a polar_stereographic CRS.

    https://proj.org/en/stable/operations/projections/stere.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()
    crs_wkt = "crs_wkt" in p

    kwargs = {
        "proj": "stere",
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    longitude_of_projection_origin = _cc_parameter(
        p, "longitude_of_projection_origin"
    )
    if longitude_of_projection_origin is not None:
        kwargs["lon_0"] = longitude_of_projection_origin
    elif not crs_wkt:
        kwargs["lon_0"] = p["straight_vertical_longitude_from_pole"]

    standard_parallel = _cc_parameter(p, "standard_parallel")
    if standard_parallel is not None:
        kwargs["lat_ts"] = standard_parallel
    elif not crs_wkt:
        kwargs["k_0"] = p["scale_factor_at_projection_origin"]

    latitude_of_projection_origin = _cc_parameter(
        p, "latitude_of_projection_origin"
    )
    if latitude_of_projection_origin is not None:
        try:
            ok = (
                latitude_of_projection_origin == -90
                or latitude_of_projection_origin == 90
            )
        except Exception:
            ok = False

        if not ok:
            if is_log_level_info(logger):
                logger.info(
                    f"Can't create coordinates for {cr!r}: "
                    "Bad 'latitude_of_projection_origin' parameter: "
                    f"{latitude_of_projection_origin!r}"
                )  # pragma: no cover

            return

        kwargs["lat_0"] = latitude_of_projection_origin

    return _create_pyproj_CRS(kwargs, cr)


def rotated_latitude_longitude(cr):
    """Create a rotated_latitude_longitude CRS.

    https://proj.org/en/stable/operations/projections/ob_tran.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "ob_tran",
        "o_proj": "longlat",
        "o_lat_p": _cc_parameter(p, "grid_north_pole_latitude"),
        "o_lon_p": _cc_parameter(p, "north_pole_grid_longitude", 0),
    }

    grid_north_pole_longitude = _cc_parameter(p, "grid_north_pole_longitude")
    if grid_north_pole_longitude is not None:
        try:
            kwargs["lon_0"] = float(grid_north_pole_longitude) + 180
        except Exception:
            if is_log_level_info(logger):
                logger.info(
                    f"Can't create coordinates for {cr!r}: "
                    "Bad 'grid_north_pole_longitude' parameter: "
                    f"{grid_north_pole_longitude!r}"
                )  # pragma: no cover

            return

    return _create_pyproj_CRS(kwargs, cr)


def sinusoidal(cr):
    """Create a sinusoidal CRS.

    https://proj.org/en/stable/operations/projections/sinu.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "sinu",
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    return _create_pyproj_CRS(kwargs, cr)


def stereographic(cr):
    """Create a stereographic CRS.

    https://proj.org/en/stable/operations/projections/stere.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "stere",
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "k_0": _cc_parameter(p, "scale_factor_at_projection_origin"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }
    return _create_pyproj_CRS(kwargs, cr)


def transverse_mercator(cr):
    """Create a transverse_mercator CRS.

    https://proj.org/en/stable/operations/projections/tmerc.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "tmerc",
        "lon_0": _cc_parameter(p, "longitude_of_central_meridian"),
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "k_0": _cc_parameter(p, "scale_factor_at_central_meridian"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }

    return _create_pyproj_CRS(kwargs, cr)


def vertical_perspective(cr):
    """Create a vertical_perspective CRS.

    https://proj.org/en/stable/operations/projections/nsper.html

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct from the CRS is deived.

    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    kwargs = {
        "proj": "nsper",
        "lat_0": _cc_parameter(p, "latitude_of_projection_origin"),
        "lon_0": _cc_parameter(p, "longitude_of_projection_origin"),
        "h": _cc_parameter(p, "perspective_point_height"),
        "x_0": _cc_parameter(p, "false_easting", 0),
        "y_0": _cc_parameter(p, "false_northing", 0),
    }
    return _create_pyproj_CRS(kwargs, cr)


def create_projection_CRS(cr, grid_mapping_name):
    """Create a projection CRS.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct that defines the
            coordinate system.

        grid_mapping_name: `str`
            The grid mapping name.

    :Returns:

        `pyproj.CRS` or `None`
            The projection CRS, or `None` if it couldn't be created.

    """
    proj = None
    try:
        match grid_mapping_name:
            case "albers_equal_area":
                proj = albers_equal_area(cr)
            case "azimuthal_equidistant":
                proj = azimuthal_equidistant(cr)
            case "geostationary":
                proj = geostationary(cr)
            case "lambert_azimuthal_equal_area":
                proj = lambert_azimuthal_equal_area(cr)
            case "lambert_conformal_conic":
                proj = lambert_conformal_conic(cr)
            case "lambert_cylindrical_equal_area":
                proj = lambert_cylindrical_equal_area(cr)
            case "latitude_longitude":
                proj = latitude_longitude(cr)
            case "mercator":
                proj = mercator(cr)
            case "oblique_mercator":
                proj = oblique_mercator(cr)
            case "orthographic":
                proj = orthographic(cr)
            case "polar_stereographic":
                proj = polar_stereographic(cr)
            case "rotated_latitude_longitude":
                proj = rotated_latitude_longitude(cr)
            case "sinusoidal":
                proj = sinusoidal(cr)
            case "stereographic":
                proj = stereographic(cr)
            case "transverse_mercator":
                proj = transverse_mercator(cr)
            case "vertical_perspective":
                proj = vertical_perspective(cr)
    except KeyError as error:
        # Trap a KeyError arising from a missing mandatory coordinate
        # reference parameter
        if is_log_level_info(logger):
            logger.info(
                f"{cr!r} has missing coordinate conversion property: {error}"
            )  # pragma: no cover

    return proj
