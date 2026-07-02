"""Utilities for creating 2-d latitude/longitude coordinates."""

import logging

import numpy as np
from cfdm import is_log_level_info

from .grid_mapping import (
    albers_equal_area,
    azimuthal_equidistant,
    geostationary,
    lambert_azimuthal_equal_area,
    lambert_conformal_conic,
    lambert_cylindrical_equal_area,
    latitude_longitude,
    mercator,
    oblique_mercator,
    orthographic,
    polar_stereographic,
    rotated_latitude_longitude,
    sinusoidal,
    stereographic,
    transverse_mercator,
    vertical_perspective,
)

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
        # Invalid non-latitude_longitude coordinate reference
        return (None, None)

    # ----------------------------------------------------------------
    # Get the source 1-d grid coordinates and axes
    # ----------------------------------------------------------------
    one_d = _get_1d_coordinates(f, cr, grid_mapping_name)
    if one_d is None:
        # Invalid 1-d grid coordinates
        return (None, None)

    # ----------------------------------------------------------------
    # Create the source grid mapping pyproj CRS
    # ----------------------------------------------------------------
    match grid_mapping_name:
        case "albers_equal_area":
            proj_src = albers_equal_area(cr)
        case "azimuthal_equidistant":
            proj_src = azimuthal_equidistant(cr)
        case "geostationary":
            proj_src = geostationary(cr)
        case "lambert_azimuthal_equal_area":
            proj_src = lambert_azimuthal_equal_area(cr)
        case "lambert_conformal_conic":
            proj_src = lambert_conformal_conic(cr)
        case "lambert_cylindrical_equal_area":
            proj_src = lambert_cylindrical_equal_area(cr)
        case "mercator":
            proj_src = mercator(cr)
        case "oblique_mercator":
            proj_src = oblique_mercator(cr)
        case "orthographic":
            proj_src = orthographic(cr)
        case "polar_stereographic":
            proj_src = polar_stereographic(cr)
        case "rotated_latitude_longitude":
            proj_src = rotated_latitude_longitude(cr)
        case "sinusoidal":
            proj_src = sinusoidal(cr)
        case "stereographic":
            proj_src = stereographic(cr)
        case "transverse_mercator":
            proj_src = transverse_mercator(cr)
        case "vertical_perspective":
            proj_src = vertical_perspective(cr)
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
    proj_latlon = latitude_longitude(cr_latlon)
    if proj_latlon is None:
        # Invalid latitude_longitude coordinate reference
        return (None, None)

    # ----------------------------------------------------------------
    # Create the 2-d lat/lon coordinates from 1-d grid coordinates
    # ----------------------------------------------------------------
    x = one_d["x"]
    y = one_d["y"]
    x = x.to_units("m")
    y = x.to_units("m")

    # Create x and y 2-d meshes of cell centres
    x_mesh, y_mesh = np.meshgrid(x.array, y.array)

    transformer = pyproj.Transformer.from_crs(
        proj_src, proj_latlon, always_xy=True, errcheck=True, radians=False
    )
    lon, lat = transformer.transform(x_mesh, y_mesh)

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
        xb = np.append(xb[:, 0], xb[-1, 1])
        yb = np.append(yb[:, 0], yb[-1, 1])

        # Create x and y 2-d meshes of unique vertices
        x_mesh, y_mesh = np.meshgrid(xb, yb)
        del xb, yb

        lon_vertices, lat_vertices = transformer.transform(x_mesh, y_mesh)

        shape = (y.size, x.size, 4)
        lat_bounds = np.empty(shape, dtype=lat_vertices.dtype)
        lon_bounds = np.empty(shape, dtype=lon_vertices.dtype)

        lat_bounds[..., 0] = lat_vertices[:-1, :-1]
        lon_bounds[..., 0] = lon_vertices[:-1, :-1]

        lat_bounds[..., 1] = lat_vertices[1:, :-1]
        lon_bounds[..., 1] = lon_vertices[1:, :-1]

        lat_bounds[..., 2] = lat_vertices[1:, 1:]
        lon_bounds[..., 2] = lon_vertices[1:, 1:]

        lat_bounds[..., 3] = lat_vertices[:-1, 1:]
        lon_bounds[..., 3] = lon_vertices[:-1, 1:]

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


def _get_1d_coordinates(f, cr, grid_mapping_name):
    """Get 1-d coordinates and axes.

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain containing the 1-d coordinates.

        cr: `CoordinateReference`
            The coordinate reference construct that implies the 1-d
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
    x = None
    y = None

    # Look for 1-d coordinates named by the coordinate reference
    for key in cr.coordinates():
        c = f.dimension_construct(f"key%{key}", default=None)
        if c is None:
            continue

        if c.X:
            key_x = key
            x = c
        elif c.Y:
            key_y = key
            y = c

    if x is None and y is None:
        # Look for 1-d coordinates by identity
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
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d latitude and longitude coordinates "
                f"for {cr!r}: Missing 1-d dimension coordinates"
            )  # pragma: no cover

        return

    return {
        "x": x,
        "y": y,
        "axis_x": f.get_data_axes(key_x)[0],
        "axis_y": f.get_data_axes(key_y)[0],
    }
