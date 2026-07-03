"""Utilities for creating 2-d latitude/longitude coordinates."""

import logging

import numpy as np
from cfdm import is_log_level_info

from cf import Units

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


def create_2d_latlon_coordinates(f, cr, cr_latlon=None, cache=True):
    """Create 2-d latitude and longitude coordinates and bounds.

    When it is not possible to create latitude and longitude
    coordinates, the reason why will be reported if the log level is
    at ``2``/``'INFO'`` or higher.

    See CF Appendix F: Grid Mappings.
    https://doi.org/10.5281/zenodo.14274886

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
            constructs, in that order; or two `None`s if the 2-d
            coordinates could not be created.

    """
    try:
        import pyproj
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 2-d lat/lon coordinates: "
                "Must install the 'pyproj' library"
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
    proj_src = _create_projection_CRS(cr, grid_mapping_name)
    if proj_src is None:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 2-d lat/lon coordinates: "
                f"Unable to create a pyproj.CRS object for {cr!r}"
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

    metres = Units("m")
    if x.Units.equivalent(metres):
        x = x.to_units(metres)

    if y.Units.equivalent(metres):
        y = y.to_units(metres)

    # Create x and y meshes of cell centres
    x_mesh, y_mesh = np.meshgrid(x.array, y.array)

    transformer = pyproj.Transformer.from_crs(
        proj_src, proj_latlon, always_xy=True
    )
    lon, lat = transformer.transform(
        x_mesh, y_mesh, errcheck=True, radians=False
    )
    del x_mesh, y_mesh
    
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

        # Create x and y meshes of vertices.
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

        lon_bounds, lat_bounds = transformer.transform(x_mesh, y_mesh)
        del x_mesh, y_mesh

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

###


def create_1d_projection_coordinates(f, cr, cr_latlon=None, cache=True):
    """Create TODO-d latitude and longitude coordinates and bounds.

    When it is not possible to create latitude and longitude
    coordinates, the reason why will be reported if the log level is
    at ``2``/``'INFO'`` or higher.

    See CF Appendix F: Grid Mappings.
    https://doi.org/10.5281/zenodo.14274886

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
            constructs, in that order; or two `None`s if the 2-d
            coordinates could not be created.

    """
    try:
        import pyproj
    except Exception:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 1-d projection coordinates: "
                "Must install the 'pyproj' library"
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
    two_d = _get_2d_latlon_coordinates(f, cr, cr_latlon)
    if two_d is None:
        # Invalid 2-d lat/lon coordinates
        return (None, None)

    # ----------------------------------------------------------------
    # Create the destination grid mapping `pyproj.CRS`
    # ----------------------------------------------------------------
    proj_dst = _create_projection_CRS(cr, grid_mapping_name)
    if proj_dst is None:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 1-d projection coordinates. "
                f"Unable to create a pyproj.CRS object for {cr!r}"
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
    transformer = pyproj.Transformer.from_crs(
        proj_latlon, proj_dst, always_xy=True
    )
    x, y = transformer.transform(
        two_d['lon'].array, two_d['lat'].array, errcheck=True, radians=False
    )

    match grid_mapping_name:
        case "rotated_latitude_longitude":
            standard_name_x = "grid_longitude"
            standard_name_y = "grid_latitude"
            units = "degrees"
        case _:
            standard_name_x = "projection_x_coordinate"
            standard_name_y = "projection_y_coordinate"
            units = "m"
            
    if not np.allclose(x, x[0]):
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 1-d projection coordinates for {cr!r}: "
                f"{standard_name_x} coordinates are not logically 1-d"
            )  # pragma: no cover

        return (None, None)

    x = x[0]
    
    if not np.allclose(y, y[:, :1]):
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 1-d projection coordinates for {cr!r}: "
                f"{standard_name_y} coordinates are not logically 1-d"
            )  # pragma: no cover
            
        return (None, None)

    y = y[:, 0]
              
    x = f._Data(x, units)
    y = f._Data(y, units)

    lon_bounds = lon.get_bounds_data(None)
    lat_bounds = lat.get_bounds_data(None)
    if lon_bounds is None or lat_bounds is None:
        x_bounds = None
        y_bounds = None
    else:
        x_bounds, y_bounds = transformer.transform(
            lon_bounds.array, lat_bounds.array, errcheck=True, radians=False
        )
 
        if not np.allclose(x_bounds, x_bounds[0]):
            x_bounds = None
            if is_log_level_info(logger):
                logger.info(
                    f"Can't create 1-d projection coordinates for {cr!r}: "
                    f"{standard_name_x} coordinates are not logically 1-d"
                )  # pragma: no cover
        else:
            x_bounds = x_bounds[0, :, 1:3]

        if not np.allclose(y_bounds, y_bounds[:, :1]):
            y_bounds = None
            if is_log_level_info(logger):
                logger.info(
                    f"Can't create 1-d projection coordinates for {cr!r}: "
                    f"{standard_name_y} coordinates are not logically 1-d"
                )  # pragma: no cover
        else:
            y_bounds = y_bounds[:, 0, :2]

        if x_bounds is not None and y_bounds is not None:            
            x_bounds = f._Bounds(data=f._Data(x_bounds))
            y_bounds = f._Bounds(data=f._Data(y_bounds))
            
    x = f._DimensionCoordinate(
        data=x,
        bounds=x_bounds,
        properties={"axis": "X", "standard_name": standard_name_x},
    )

    y = f._DimensionCoordinate(
        data=y,
        bounds=y_bounds,
        properties={"axis": "Y", "standard_name": standard_name_y},
    )

def _get_1d_coordinates(f, cr, grid_mapping_name):
    """Get 1-d dimension coordinates and axes.

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain containing the 1-d dimension
            coordinates.

        cr: `CoordinateReference`
            The coordinate reference construct that defines or implies
            the 1-d dimension coordinates.

        grid_mapping_name: `str`
            The grid_mapping_name parameter of *cr*.

    :Returns:

        `dict` or `None`
            The 1-d coordinates and axes in the following dictionary
            keys:

            * ``'x'``: The X coordinate construct
            * ``'y'``: The Y coordinate construct
            * ``'axis_x'``: The X domain axis construct key
            * ``'axis_y'``: The Y domain axis construct key

            If both 1-d dimension coordinates could not be found then
            `None` is returned.

    """
    x = None
    y = None

    # Look for 1-d coordinates named by the coordinate reference
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
                f"Can't create 2-d lat/lon coordinates for {cr!r}: "
                "Missing 1-d dimension coordinates"
            )  # pragma: no cover

        return

    # Make sure the 1-d coordinates are referenced from the coordinate
    # reference
    cr.set_coordinates((key_x, key_y))

    return {
        "x": x,
        "y": y,
        "axis_x": f.get_data_axes(key_x)[0],
        "axis_y": f.get_data_axes(key_y)[0],
    }

def _get_2d_latlon_coordinates(f, cr, cr_latlon):
    """TODO"""
    for ref in (cr_latlon, cr):
        lat = None
        lon = None        

        if ref is None:
            continue
        
        for key in ref.coordinates():
            ac = f.auxiliary_coordinate(f"key%{key}", default=None)
            if ac is None:
                continue
            
            if ac.ndim != 2:
                continue
            
            if ac.Units.islongitude:
                key_lon = key
                lon  = ac
            elif ac.Units.islatitude:
                key_lat = key
                lat = ac
                
        if lon is not  None and lat is not None:
            break
        
        
    if lon is None and lat is None:    
        key_lon, lon = f.auxiliary_coordinate(
            'X', filter_by_naxes=(2,), item=True, default=(None, None)
        )
        key_lat, lat = f.auxiliary_coordinate(
            'Y', filter_by_naxes=(2,), item=True, default=(None, None)
        )

    if lat is None or lat.ndim !=2 or not lat.Units.islatitude:        
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 1-d projection coordinates for {cr!r}: "
                "Missing 2-d latitude coordinates"
            )  # pragma: no cover

        return
        
    if lon is None or lon.ndim !=2 or not lon.Units.islongitude:        
        if is_log_level_info(logger):
            logger.info(
                f"Can't create 1-d projection coordinates for {cr!r}: "
                "Missing 2-d longitude coordinates"
            )  # pragma: no cover

        return

    axes_lat = f.get_data_axes(key_lat)
    axes_lon = f.get_data_axes(key_lon)
    if axes_lon != axes_lat:
        axes_lon = axes_lon[::-1]
        if axes_lon != axes_lat:
            if is_log_level_info(logger):
                logger.info(
                    f"Can't create 1-d projection coordinates for {cr!r}: "
                    "2-d lat/lon coordinates span different axes"
                )  # pragma: no cover
                
            return

        lon = lon.transpose()

    return {'lat': lat, 'lon': lon, 'axes': axes_lat}

def _create_projection_CRS(cr, grid_mapping_name):
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
        case _:
            proj = None

    return proj
