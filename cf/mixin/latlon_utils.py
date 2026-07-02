"""2-d latitude/longitude coordinates functionality."""

import logging

import numpy as np
from cfdm import is_log_level_info

logger = logging.getLogger(__name__)


def _create_2d_latlon_coordinates(f, cr, cr_latlon=None, cache=True):
    """Create 2-d latitude and longitude coordinates and bounds.

    When it is not possible to create latitude and longitude
    coordinates, the reason why will be reported if the log level is
    at ``2``/``'INFO'`` or higher.

    See CF Appendix F: Grid Mappings.
    https://doi.org/10.5281/zenodo.14274886

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain containing the ??? grid, which will be
            updated in-place.

        cr: `CoordinateReference`
            The coordinate reference construct for the
            non-latitude_longitude grid mapping.

        cr_latlon: `CoordinateReference` or `None`
            The coordinate reference construct for the
            latitude_longitude grid mapping, or `None` is there isn't
            one.

        cache: `bool`, optional
            If True (the default) then cache in memory the first and
            last of any newly-created coordinates and bounds. This may
            slightly slow down the coordinate creation process, but
            may greatly speed up, and reduce the memory requirement
            of, a future inspection of the coordinates and
            bounds. Even when *cache* is True, new cached coordinate
            values can only be created if the existing 1-d coordinates
            themselves have cached first and last values.

    :Returns:

        (`str`, `str`) or (`None`, `None`)
            The keys of the new 2-d latitude and longitude coordinate
            constructs, in that order, or two `None`s if the 2-d
            coordinates could not be created.

    """
    try:
        import pyproj
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d latitude and longitude coordinates "
                f"for {cr!r}: Must install the 'pyproj' library"
            )  # pragma: no cover

        return (None, None)

    grid_mapping_name = cr.coordinate_conversion.get_parameter(
        "grid_mapping_name", None
    )
    if grid_mapping_name is None:
        return (None, None)

    # ----------------------------------------------------------------
    # Get the source 1-d grid coordinates and axes
    # ----------------------------------------------------------------
    one_d = _get_1d_coordinates(f, cr, grid_mapping_name)
    if one_d is None:
        return (None, None)

    # ----------------------------------------------------------------
    # Create the source grid mapping pyproj CRS
    # ----------------------------------------------------------------
    match grid_mapping_name:
        case "albers_equal_area":
            proj_src = _albers_equal_area(cr)
        case "azimuthal_equidistant":
            proj_src = _azimuthal_equidistant(cr)
        case "geostationary":
            proj_src = _geostationary(cr)
        case "lambert_azimuthal_equal_area":
            proj_src = _lambert_azimuthal_equal_area(cr)
        case "lambert_conformal_conic":
            proj_src = _lambert_conformal_conic(cr)
        case "lambert_cylindrical_equal_area":
            proj_src = _lambert_cylindrical_equal_area(cr)
        case "mercator":
            proj_src = _mercator(cr)
        case "oblique_mercator":
            proj_src = _oblique_mercator(cr)
        case "orthographic":
            proj_src = _orthographic(cr)
        case "polar_stereographic":
            proj_src = _polar_stereographic(cr)
        case "rotated_latitude_longitude":
            proj_src = _rotated_latitude_longitude(cr)
        case "sinusoidal":
            proj_src = _sinusoidal(cr)
        case "stereographic":
            proj_src = _stereographic(cr)
        case "transverse_mercator":
            proj_src = _transverse_mercator(cr)
        case "vertical_perspective":
            proj_src = _vertical_perspective(cr)
        case _:
            if is_log_level_info(logger):
                logger.info(
                    "Can't create 2-d latitude and longitude coordinates "
                    f"for {cr!r}"
                )  # pragma: no cover

            return (None, None)

    if proj_src is None:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d latitude and longitude coordinates. "
                f"Unable to create a pyproj.CRS object for {cr!r} from "
                f"the grid mapping parameters: "
                f"{cr.coordinate_conversion.parameters()!r}"
            )  # pragma: no cover

        return (None, None)

    # ----------------------------------------------------------------
    # Create the target latitude_longitude pyproj CRS
    # ----------------------------------------------------------------
    proj_latlon = _latitude_longitude(cr_latlon)
    if proj_latlon is None:
        return (None, None)

    # ----------------------------------------------------------------
    # Create the 2-d lat/lon coordinates from 1-d grid coordinates
    # ----------------------------------------------------------------
    x = one_d["x"]
    y = one_d["y"]
    x = x.to_units('m')
    y = x.to_units('m')
    x_mesh, y_mesh = np.meshgrid(x.array, y.array)

    transformer = pyproj.Transformer.from_crs(
        proj_src, proj_latlon, always_xy=True, errcheck=True, radians=False
    )
    lon_2d, lat_2d = transformer.transform(x_mesh, y_mesh)

    # ----------------------------------------------------------------
    # Create the 2-d lat/lon bounds from 1-d grid coordinate bounds
    # ----------------------------------------------------------------
    xb = x.get_bounds_data(None)
    yb = y.get_bounds_data(None)
    if xb is None and yb is None:
        lat_2d_bounds = None
        lon_2d_bounds = None
    else:
        xb = xb.array
        yb = yb.array
        xb = np.append(xb[:, 0], xb[-1, 1])
        yb = np.append(yb[:, 0], yb[-1, 1])

        x_mesh, y_mesh = np.meshgrid(xb, yb)
        del xb, yb

        lon_2d_vertices, lat_2d_vertices = transformer.transform(
            x_mesh, y_mesh
        )

        shape = (y.size, x.size, 4)
        lat_2d_bounds = np.empty(shape, dtype=lat_2d.dtype)
        lon_2d_bounds = np.empty(shape, dtype=lon_2d.dtype)

        lat_2d_bounds[..., 0] = lat_2d_vertices[:-1, :-1]
        lon_2d_bounds[..., 0] = lon_2d_vertices[:-1, :-1]

        lat_2d_bounds[..., 1] = lat_2d_vertices[1:, :-1]
        lon_2d_bounds[..., 1] = lon_2d_vertices[1:, :-1]

        lat_2d_bounds[..., 2] = lat_2d_vertices[1:, 1:]
        lon_2d_bounds[..., 2] = lon_2d_vertices[1:, 1:]

        lat_2d_bounds[..., 3] = lat_2d_vertices[:-1, 1:]
        lon_2d_bounds[..., 3] = lon_2d_vertices[:-1, 1:]

        lat_2d_bounds = f._Bounds(data=f._Data(lat_2d_bounds))
        lon_2d_bounds = f._Bounds(data=f._Data(lon_2d_bounds))

    # ----------------------------------------------------------------
    # Add the 2-d lat/lon coordinates to the domain
    # ----------------------------------------------------------------
    lat_2d = f._AuxiliaryCoordinate(
        data=f._Data(lat_2d, "degrees_north"),
        bounds=lat_2d_bounds,
        properties={"standard_name": "latitude"},
    )
    lon_2d = f._AuxiliaryCoordinate(
        data=f._Data(lon_2d, "degrees_east"),
        bounds=lon_2d_bounds,
        properties={"standard_name": "longitude"},
    )

    axes = (one_d["axis_y"], one_d["axis_x"])

    lat_key = f.set_construct(lat_2d, axes=axes, copy=False)
    lon_key = f.set_construct(lon_2d, axes=axes, copy=False)

    return (lat_key, lon_key)


def _create_proj_CRS(kwargs, cr):
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

    # Create the pyproj.CRS keywword arguments, which include
    # parameters for describing the ellipsoid
    kwargs = _get_ellipsoid_parameters(cr) | kwargs
    
    # Remove `None` values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        proj = pyproj.CRS(**kwargs)
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d latitude and longitude coordinates "
                f"for {cr!r}: Bad pyproj.CRS parameters: {kwargs}"
            )  # pragma: no cover
            
        return

    return proj

def _get_ellipsoid_parameters(cr):
    """TODO"""
    kwargs = {}
    if cr is None:
        return kwargs
    
    parameters = cr_latlon.coordinate_conversion.parameters()
    
    if "earth_radius" in parameters:
        kwargs["R"] = parameters.get("earth_radius")
    elif "semi_major_axis" in parameters:
        kwargs["a"] = parameters.get("semi_major_axis")
        kwargs["rf"] = parameters.get("inverse_flattening")
        kwargs["b"] = parameters.get("semi_minor_axis")
    elif "reference_ellipsoid_name" in parameters:
        kwargs["ellps"] = parameters.get("reference_ellipsoid_name")
    else:
        kwargs["ellps"] = "sphere"
        
    if "longitude_of_prime_meridian" in parameters:
        kwargs["pm"] = parameters.get("longitude_of_prime_meridian", 0)
    elif "prime_meridian_name" in parameters:
        kwargs["pm"] = parameters.get("prime_meridian_name")

    return kwargs
            

def _create_latitude_longitude_CRS(cr_latlon):
    """Create a latitude_longitude `pyproj.CRS` instance.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr_latlon: `CoordinateReference` or `None`
            The latitude_longitude coordinate reference construct from
            which to create the CRS, or `None` if there isn't one.

    :Returns:

        `pyproj.CRS` or `None`
            The created CRS, or `None` if one couldn't be created.

    """
    kwargs = {"proj": "longlat"}
    if cr_latlon is None:
        kwargs["ellps"] = "sphere"

    return _create_proj_CRS(kwargs, cr_latlon)


def _get_1d_coordinates(f, cr, grid_mapping_name):
    """Get 1-d coordinates and axes.

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain containing the 1-d coordinates.

        cr: `CoordinateReference`
            The coordinate reference construct that references the 1-d
            coordinates.

        grid_mapping_name: `str`
            The grid_mapping_name parameter of *cr*.

    :Returns:

        `dict`

            The 1-d coordinates and axes in the following dictionary
            keys:

            * ``'x'``: The X coordinate construct
            * ``'y'``: The Y coordinate construct
            * ``'axis_x'``: The X domain axis construct key
            * ``'axis_y'``: The Y domain axis construct key

    """
    match grid_mapping_name:
        case "rotated_latitude_longitude":
            identity_x = "grid_longitude"
            identity_y = "grid_latitude"
        case _:
            identity_x = "projection_x_coordinate"
            identity_y = "projection_y_coordinate"

    key_x, x = f.dimension_coordinate(
        identity_x, item=True, default=(None, None)
    )
    key_y, y = f.dimension_coordinate(
        identity_y, item=True, default=(None, None)
    )

    if x is None and is_log_level_info(logger):
        logger.info(
            "Can't create 2-d latitude and longitude coordinates "
            f"for {cr!r}: Missing 1-d {identity_x!r} dimension coordinates"
        )  # pragma: no cover
        return

    if y is None and is_log_level_info(logger):
        logger.info(
            "Can't create 2-d latitude and longitude coordinates "
            f"for {cr!r}: Missing 1-d {identity_y!r} dimension coordinates"
        )  # pragma: no cover
        return

    return {
        "x": x,
        "y": y,
        "axis_x": f.get_data_axes(key_x)[0],
        "axis_y": f.get_data_axes(key_y)[0],
    }


# ====================================================================
# Functions for creating `pyproj.CRS` instances for each grid mapping
#
# These functions are called by `_create_2d_latlon_coordinates`
# ====================================================================

def _albers_equal_area(cr):
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
        lat_1 =     standard_parallel
    else:
        try:            
            lat_2 = standard_parallel[1]
        except Exception:
            pass

    kwargs['lat_1'] = lat_1
    kwargs['lat_2'] = lat_2

    return _create_proj_CRS(kwargs, cr)


def _azimuthal_equidistant(cr):
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

    return _create_proj_CRS(kwargs, cr)


def _geostationary(cr):
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
    kwargs = 
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
        case 'x':
            ok = fixed_angle_axis in (None, 'y')
        case 'y':
            ok = fixed_angle_axis in (None, 'x')
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
    
    return _create_proj_CRS(kwargs, cr)

def _lambert_azimuthal_equal_area(cr):
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
    return _create_proj_CRS(kwargs, cr)


def _lambert_conformal_conic(cr):
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
        lat_1 =     standard_parallel
    else:
        try:            
            lat_2 = standard_parallel[1]
        except Exception:
            pass

    kwargs['lat_1'] = lat_1
    kwargs['lat_2'] = lat_2

    return _create_proj_CRS(kwargs, cr)

def _lambert_cylindrical_equal_area(cr):
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
        
    return _create_proj_CRS(kwargs, cr)


def _latitude_longitude(cr):
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
    if cr is None:
        kwargs["ellps"] = "sphere"
        
    return _create_proj_CRS(kwargs, cr)


def _mercator(cr):
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

    return _create_proj_CRS(kwargs, cr)


def _oblique_mercator(cr):
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
    return _create_proj_CRS(kwargs, cr)


def _orthographic(cr):
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
    return _create_proj_CRS(kwargs, cr)


def _polar_stereographic(cr):
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
        ok = latitude_of_projection_origin == -90 or latitude_of_projection_origin == 90
    except Exception:
        ok = False
        
    if not ok:
        logger.info(
            "Can't create 2-d latitude and longitude coordinates "
            f"for {cr!r}: Bad 'latitude_of_projection_origin' parameter: "
            f"{latitude_of_projection_origin!r}"
        )  # pragma: no cover
        
    kwargs["lat_0"] = latitude_of_projection_origin

    return _create_proj_CRS(kwargs, cr)

def _rotated_latitude_longitude(cr):
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

    return _create_proj_CRS(kwargs, cr)

def _sinusoidal(cr):
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

    return _create_proj_CRS(kwargs, cr)


def _stereographic(cr):
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
    return _create_proj_CRS(kwargs, cr)

def _transverse_mercator(cr):
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
        "k_0":   p.get("scale_factor_at_central_meridian"),
        "x_0":   p.get("false_easting", 0),
        "y_0":   p.get("false_northing", 0),
    }

    return _create_proj_CRS(kwargs, cr)


def _vertical_perspective(cr):
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
    return _create_proj_CRS(kwargs, cr)
##########################

def _albers_equal_area(cr):
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
        lat_1 =     standard_parallel
    else:
        try:            
            lat_2 = standard_parallel[1]
        except Exception:
            pass

    kwargs['lat_1'] = lat_1
    kwargs['lat_2'] = lat_2

    return _create_proj_CRS(kwargs, cr)


def _azimuthal_equidistant(cr):
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

    return _create_proj_CRS(kwargs, cr)


def _geostationary(cr):
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
    kwargs = 
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
        case 'x':
            ok = fixed_angle_axis in (None, 'y')
        case 'y':
            ok = fixed_angle_axis in (None, 'x')
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
    
    return _create_proj_CRS(kwargs, cr)

def _lambert_azimuthal_equal_area(cr):
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
    return _create_proj_CRS(kwargs, cr)


def _lambert_conformal_conic(cr):
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
        lat_1 =     standard_parallel
    else:
        try:            
            lat_2 = standard_parallel[1]
        except Exception:
            pass

    kwargs['lat_1'] = lat_1
    kwargs['lat_2'] = lat_2

    return _create_proj_CRS(kwargs, cr)

def _lambert_cylindrical_equal_area(cr):
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
        
    return _create_proj_CRS(kwargs, cr)


def _latitude_longitude(cr):
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
    if cr is None:
        kwargs["ellps"] = "sphere"
        
    return _create_proj_CRS(kwargs, cr)


def _mercator(cr):
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

    return _create_proj_CRS(kwargs, cr)


def _oblique_mercator(cr):
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
    return _create_proj_CRS(kwargs, cr)


def _orthographic(cr):
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
    return _create_proj_CRS(kwargs, cr)


def _polar_stereographic(cr):
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
        ok = latitude_of_projection_origin == -90 or latitude_of_projection_origin == 90
    except Exception:
        ok = False
        
    if not ok:
        logger.info(
            "Can't create 2-d latitude and longitude coordinates "
            f"for {cr!r}: Bad 'latitude_of_projection_origin' parameter: "
            f"{latitude_of_projection_origin!r}"
        )  # pragma: no cover
        
    kwargs["lat_0"] = latitude_of_projection_origin

    return _create_proj_CRS(kwargs, cr)

def _rotated_latitude_longitude(cr):
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

    return _create_proj_CRS(kwargs, cr)

def _sinusoidal(cr):
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

    return _create_proj_CRS(kwargs, cr)


def _stereographic(cr):
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
    return _create_proj_CRS(kwargs, cr)

def _transverse_mercator(cr):
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
        "k_0":   p.get("scale_factor_at_central_meridian"),
        "x_0":   p.get("false_easting", 0),
        "y_0":   p.get("false_northing", 0),
    }

    return _create_proj_CRS(kwargs, cr)


def _vertical_perspective(cr):
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
    return _create_proj_CRS(kwargs, cr)

