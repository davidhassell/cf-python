"""Utilities for creating 2-d latitude/longitude coordinates."""

import logging

import numpy as np
from cfdm import is_log_level_info

from cf import Units

from .grid_mapping import create_projection_CRS

logger = logging.getLogger(__name__)


def create_2d_latlon_coordinates(f, cr, cr_latlon, longitude_at_pole=None):
    """Create 2-d latitude and longitude coordinates and bounds.

    Creates the 2-d latitude and longitude coordinate constructs that
    are implied by the coordinate reference constructs.

    When it is not possible to create latitude and longitude
    coordinates, the reason why will be reported if the log level is
    at ``2``/``'INFO'`` or higher.

    If the log level is at ``3``/``'DEBUG'``/``-1`` then a description
    of the `pyproj.CRS` instances used to create 2-d latitude and
    longitude coordinates will also be shown.

    See CF Appendix F: Grid Mappings
    (https://doi.org/10.5281/zenodo.14274886).

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain, which will be updated in-place,
            containing non-latitude_longitude grid.

        cr: `CoordinateReference`
            The coordinate reference construct for the
            non-latitude_longitude grid mapping.

        cr_latlon: `CoordinateReference` or `None`
            The coordinate reference construct for the
            latitude_longitude grid mapping, or `None` if there isn't
            one, in which case a spherical latitude_longitude grid
            mapping is assumed.

        longitude_at_pole: `None` or number
            Define the treatment of longitudes of coordinates or
            coordinate bounds that lie exactly on the north or south
            pole. If `None` (the default) then the longitudes of such
            points are determined by whichever algorithm was used to
            create the coordinates, which could result in different
            grid points on a pole having different longitudes. If set
            to a number, then the longitudes of all grid points on the
            north or south pole will be given that value.

    :Returns:

        (`str`, `str`) or (`None`, `None`)
            The keys of the new 2-d latitude and longitude coordinate
            constructs, in that order; or two `None`s if the 2-d
            coordinates could not be created.

    """
    try:
        import pyproj
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                "Must install the 'pyproj' library"
            )  # pragma: no cover

        return (None, None)

    grid_mapping_name = cr.coordinate_conversion.get_parameter(
        "grid_mapping_name", None
    )
    if grid_mapping_name is None:
        # Invalid non-latitude_longitude coordinate reference
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                f"Unable to create a {grid_mapping_name} pyproj.CRS object"
            )  # pragma: no cover

        return (None, None)

    # ----------------------------------------------------------------
    # Get the source 1-d grid coordinates and axes
    # ----------------------------------------------------------------
    one_d = _get_1d_coordinates(f, cr)
    if one_d is None:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                "Can't find all 1-d dimension coordinates"
            )  # pragma: no cover

        return (None, None)

    # ----------------------------------------------------------------
    # Create the source projection CRS
    # ----------------------------------------------------------------
    proj_src = create_projection_CRS(cr, grid_mapping_name)
    if proj_src is None:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                f"Unable to create a {grid_mapping_name} pyproj.CRS object"
            )  # pragma: no cover

        return (None, None)

    # ----------------------------------------------------------------
    # Create the destination latitude_longitude CRS
    # ----------------------------------------------------------------
    if cr_latlon is None:
        # When a specific latitude_longitude coordinate reference has
        # not been provided, get the shape of the ellipsoid from the
        # source projection coordinate reference.
        cr_latlon = cr

    proj_latlon = create_projection_CRS(cr_latlon, "latitude_longitude")
    if proj_latlon is None:
        # Invalid latitude_longitude coordinate reference
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                "Unable to create a latitude_longitude pyproj.CRS object "
                f"from {cr_latlon!r}"
            )  # pragma: no cover

        return (None, None)

    # ----------------------------------------------------------------
    # Create the transform function that converts source coordinates
    # to destination coordinates
    # ----------------------------------------------------------------
    try:
        transformer = pyproj.Transformer.from_crs(
            proj_src, proj_latlon, always_xy=True
        )
    except Exception as error:
        # Invalid latitude_longitude coordinate reference
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                f"Error during pyproj.Transformer.from_crs: {error}"
            )  # pragma: no cover

        return (None, None)

    # ----------------------------------------------------------------
    # Create 2-d lat/lon coordinates from 1-d grid coordinate centres
    # ----------------------------------------------------------------
    x = one_d["x"]
    y = one_d["y"]

    # The transform function requires that the projection coordinates
    # have units of metres
    metres = Units("m")
    if x.Units.equivalent(metres):
        x = x.to_units(metres)

    if y.Units.equivalent(metres):
        y = y.to_units(metres)

    # Create meshes of x and y cell centres
    x_mesh, y_mesh = np.meshgrid(x.array, y.array)

    try:
        lon, lat = transformer.transform(
            x_mesh, y_mesh, errcheck=True, radians=False
        )
    except Exception as error:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                f"Error during pyproj coordinate transformation: {error}"
            )  # pragma: no cover

        return (None, None)

    del x_mesh, y_mesh

    if longitude_at_pole is not None:
        # Set the longitude at the poles
        lon = np.where((lat == -90) | (lat == 90), longitude_at_pole, lon)

    lat = f._Data(lat, "degrees_north")
    lon = f._Data(lon, "degrees_east")

    # ----------------------------------------------------------------
    # Create the 2-d lat/lon bounds from 1-d grid coordinate bounds
    # ----------------------------------------------------------------
    xb = x.get_bounds_data(None)
    yb = y.get_bounds_data(None)
    if xb is None or yb is None:
        lat_bounds = None
        lon_bounds = None
    else:
        xb = xb.array
        yb = yb.array

        # Create meshes of x and y vertices.
        shape = (y.size, x.size)
        xb = np.broadcast_to(xb[np.newaxis, :, :], shape + (2,))
        yb = np.broadcast_to(yb[:, np.newaxis, :], shape + (2,))

        x_mesh = np.empty(shape + (4,), dtype=xb.dtype)
        y_mesh = np.empty(shape + (4,), dtype=yb.dtype)

        x_mesh[..., 0] = xb[..., 0]
        y_mesh[..., 0] = yb[..., 0]

        x_mesh[..., 1] = xb[..., 0]
        y_mesh[..., 1] = yb[..., 1]

        x_mesh[..., 2] = xb[..., 1]
        y_mesh[..., 2] = yb[..., 1]

        x_mesh[..., 3] = xb[..., 1]
        y_mesh[..., 3] = yb[..., 0]
        del xb, yb

        try:
            lon_bounds, lat_bounds = transformer.transform(x_mesh, y_mesh)
        except Exception as error:
            if is_log_level_info(logger):
                logger.info(
                    f"Can't create 2-d lat/lon coordinate bounds for {cr!r}: "
                    f"Error during pyproj transformation: {error}"
                )  # pragma: no cover

            return (None, None)

        del x_mesh, y_mesh

        if longitude_at_pole is not None:
            # Set the longitude at the poles
            lon_bounds = np.where(
                (lat_bounds == -90) | (lat_bounds == 90),
                longitude_at_pole,
                lon_bounds,
            )

        lat_bounds = f._Bounds(data=f._Data(lat_bounds))
        lon_bounds = f._Bounds(data=f._Data(lon_bounds))

    # ----------------------------------------------------------------
    # Add the 2-d lat/lon coordinates to the domain
    # ----------------------------------------------------------------
    aux_lat = f._AuxiliaryCoordinate(
        data=lat,
        bounds=lat_bounds,
        properties={"standard_name": "latitude"},
    )
    aux_lon = f._AuxiliaryCoordinate(
        data=lon,
        bounds=lon_bounds,
        properties={"standard_name": "longitude"},
    )

    axes = (one_d["axis_y"], one_d["axis_x"])

    lat_key = f.set_construct(aux_lat, axes=axes, copy=False)
    lon_key = f.set_construct(aux_lon, axes=axes, copy=False)

    return (lat_key, lon_key)


def _get_1d_coordinates(f, cr):
    """Get 1-d dimension coordinates and axes.

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain containing the 1-d dimension
            coordinates.

        cr: `CoordinateReference`
            The coordinate reference construct that defines or implies
            the 1-d dimension coordinates.

    :Returns:

        `dict` or `None`
            The 1-d coordinates and axes in the following dictionary
            keys:

            * ``'x'``: The X dimension coordinate construct.
            * ``'y'``: The Y dimension coordinate construct.
            * ``'axis_x'``: The X domain axis construct key.
            * ``'axis_y'``: The Y domain axis construct key.

            If both 1-d dimension coordinates could not be found then
            `None` is returned.

    **Examples:**

    >>> _get_1d_coordinates(f, cr)
    {'x': <CF DimensionCoordinate: projection_x_coordinate(10) km>,
     'y': <CF DimensionCoordinate: projection_y_coordinate(12) km>,
     'axis_x': 'domainaxis1',
     'axis_y': 'domainaxis0'}

    >>> _get_1d_coordinates(f, cr)
    {'x': <CF DimensionCoordinate: grid_longitude(106) degrees>,
     'y': <CF DimensionCoordinate: grid_latitude(111) degrees>,
     'axis_x': 'domainaxis1',
     'axis_y': 'domainaxis0'}

    """
    x = None
    y = None

    # Look for dimension coordinates named by the coordinate reference
    for key in cr.coordinates():
        dc = f.dimension_coordinate(f"key%{key}", default=None)
        if dc is None:
            continue

        if dc.X:
            key_x = key
            x = dc
        elif dc.Y:
            key_y = key
            y = dc

    if x is None and y is None:
        # Look for 1-d coordinates by identity
        grid_mapping_name = cr.coordinate_conversion.get_parameter(
            "grid_mapping_name", None
        )
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

    if x is None or y is None:
        # Can't find both 1-d dimension coordinates
        return

    # Make sure that the 1-d coordinates are referenced from the
    # coordinate reference
    cr.set_coordinates((key_x, key_y))

    return {
        "x": x,
        "y": y,
        "axis_x": f.get_data_axes(key_x)[0],
        "axis_y": f.get_data_axes(key_y)[0],
    }
