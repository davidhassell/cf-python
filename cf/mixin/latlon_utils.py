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
        case "rotated_latitude_longitude":
            proj_src = _rotated_latitude_longitude(cr)
        case "healpix" | "reduced_gaussian":
            raise ValueError(
                "Can't create 2-d latitude and longitude coordinates "
                f"for {cr!r}"
            )
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
    proj_latlon = _create_latitude_longitude_CRS(cr_latlon)
    if proj_latlon is None:
        return (None, None)

    # ----------------------------------------------------------------
    # Create the 2-d lat/lon coordinates from 1-d grid coordinates
    # ----------------------------------------------------------------
    x = one_d["x"]
    y = one_d["y"]
    lon_2d_mesh, lat_2d_mesh = np.meshgrid(x.array, y.array)

    transformer = pyproj.Transformer.from_crs(
        proj_src, proj_latlon, always_xy=True
    )
    lon_2d, lat_2d = transformer.transform(lon_2d_mesh, lat_2d_mesh)

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

        lon_2d_mesh, lat_2d_mesh = np.meshgrid(xb, yb)

        lon_2d_vertices, lat_2d_vertices = transformer.transform(
            lon_2d_mesh, lat_2d_mesh
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


def _rotated_latitude_longitude(cr):
    """Create a rotated_latitude_longitude `pyproj.CRS` instance.

    .. versionadded:: NEXTVERSION

    :Parameters:

        cr: `CoordinateReference`
            The coordinate reference construct.
    
    :Returns:

        `pyproj.CRS`
            The created CRS, or `None` if one couldn't be created.

    """
    p = cr.coordinate_conversion.parameters()

    pole_lon = p.get("grid_north_pole_longitude")
    try:
        pole_lon = float(pole_lon)
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d latitude and longitude coordinates "
                f"for {cr!r}: Bad 'grid_north_pole_longitude' parameter: "
                f"{pole_lon!r}"
            )  # pragma: no cover

        return

    kwargs = {
        "proj": "ob_tran",
        "o_proj": "longlat",
        "o_lon_p": p.get("north_pole_grid_longitude", 0),
        "o_lat_p": p.get("grid_north_pole_latitude"),
        "lon_0": pole_lon + 180,
    }
    proj = _create_proj_CRS(kwargs, cr)

    return proj

def _transverse_mercator(cr):
    """Create a transerve_mercator `pyproj.CRS` instance.

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
        "x_0":   p.get("false_easting"),
        "y_0":   p.get("false_northing"),
    }

    return _create_proj_CRS(kwargs, cr)

#---------------


def _albers_equal_area(cr):
    """Create a albers_equal_area `pyproj.CRS` instance.

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
        "x_0": p.get("false_easting"),
        "y_0": p.get("false_northing"),
    }

    standard_parallel = p.get("standard_parallel")
    try:
        lat_1 = standard_parallel[0]
    except Exception:
        lat_1 =     standard_parallel
    else:
        
    kwargs = {
        "lat_1": p.get("standard_parallel")[0] if isinstance(p.get("standard_parallel"), (list, tuple)) else p.get("standard_parallel"),
        "lat_2": p.get("standard_parallel")[1] if isinstance(p.get("standard_parallel"), (list, tuple)) and len(p.get("standard_parallel")) > 1 else None,
    }

    return _create_proj_CRS(kwargs)


def _azimuthal_equidistant(cr):
    """Create a `pyproj.CRS` instance for Azimuthal Equidistant."""
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "aeqd",
        "lat_0": p.get("latitude_of_projection_origin"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "x_0": p.get("false_easting"),
        "y_0": p.get("false_northing"),
    }
    kwargs.update(_extract_datum_parameters(cr))
    return _create_proj_CRS({k: v for k, v in kwargs.items() if v is not None}, cr)


def _geostationary(cr):
    """Create a `pyproj.CRS` instance for Geostationary Satellite."""
    p = cr.coordinate_conversion.parameters()
    kwargs = {
        "proj": "geos",
        "h": p.get("perspective_point_height"),
        "lon_0": p.get("longitude_of_projection_origin"),
        "sweep": p.get("sweep_angle_axis"),
        "x_0": p.get("false_easting"),
        "y_0": p.get("false_northing"),
    }
    kwargs.update(_extract_datum_parameters(cr))
    return _create_proj_CRS({k: v for k, v in kwargs.items() if v is not None}, cr)
