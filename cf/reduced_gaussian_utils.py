   case "grid_mapping_name:reduced_gaussian":
                    # ------------------------------------------------
                    # Reduced Gaussian
                    # ------------------------------------------------
                    if is_log_level_info(logger):
                        logger.info(
                            "Can't yet create latitude and longitude "
                            f"coordinates from {cr!r}"
                        )  # pragma: no cover

                        import numpy as np

def calculate_cf_reduced_gaussian(indices, lats, pl):
    """
    Calculates latitudes, longitudes, and their bounds for a list of 1D indices
    on a Reduced Gaussian Grid following the CF conventions.
    
    Parameters:
    -----------
    indices : array-like of int
        0-based 1D indices of the grid cells to lookup.
    lats : array-like of float
        The explicit latitude vector directly read from the CF file.
    pl : array-like of int
        The points_per_latitude vector from the CF file.
        
    Returns:
    --------
    dict containing:
        - 'lat': Latitude of each index
        - 'lon': Longitude of each index
        - 'lat_bounds': Minimum and maximum latitude bounds (N, 2)
        - 'lon_bounds': Minimum and maximum longitude bounds (N, 2)
    """
    indices = np.atleast_1d(indices)
    lats = np.asarray(lats)
    pl = np.asarray(pl)
    
    # 1. Compute cumulative points per latitude line (accum_pl)
    accum_pl = np.cumsum(pl)
    
    # 2. Find the latitude row index 'k' for each 1D index.
    # np.searchsorted(..., side='right') perfectly mimics: min{s | accum_pl[s] > i}
    k = np.searchsorted(accum_pl, indices, side='right')
    
    # 3. Find the longitude column index 'm' within that row.
    # To avoid 'k-1' out-of-bounds on row 0, we prepend a 0 to the accumulator.
    prev_accum = np.zeros(len(accum_pl) + 1, dtype=accum_pl.dtype)
    prev_accum[1:] = accum_pl
    m = indices - prev_accum[k]
    
    # 4. Calculate point coordinates
    lat_i = lats[k]
    lon_i = m * (360.0 / pl[k])
    
    # 5. Calculate Bounds (Midpoints)
    # Longitude bounds are exactly +/- half the grid spacing for that row
    lon_spacing = 360.0 / pl[k]
    lon_bounds = np.column_stack((lon_i - 0.5 * lon_spacing, lon_i + 0.5 * lon_spacing))
    
    # Latitude bounds are midpoints between consecutive latitude rows
    # Prepend/append virtual bounds to handle the edges (the poles)
    extended_lats = np.zeros(len(lats) + 2)
    extended_lats[0] = 90.0 if lats[0] > 0 else -90.0   # North pole
    extended_lats[1:-1] = lats
    extended_lats[-1] = -90.0 if lats[-1] < 0 else 90.0 # South pole
    
    # Midpoints of the extended array
    midpoints = 0.5 * (extended_lats[:-1] + extended_lats[1:])
    
    # Extract the upper (north) and lower (south) limits for each k
    # Since lats go North -> South, midpoints[k] is the upper limit, midpoints[k+1] is lower
    lat_bounds = np.column_stack((midpoints[k + 1], midpoints[k]))
    
    return {
        "lat": lat_i,
        "lon": lon_i,
        "lat_bounds": lat_bounds,
        "lon_bounds": lon_bounds
    }
