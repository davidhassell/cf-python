import numpy as np

from cf import atol as cf_atol


def _ugrid_locate(y, x, f, latlon=True, unique=True, return_operator=False):
    """Locate UGRID cells containing latitude-longitude locations.

    Returns the discrete axis indices of the cells containing the
    latitude-longitude locations.

    If a single latitude is given then it is paired with each
    longitude, and if a single longitude is given then it is paired
    with each latitude. If multiple latitudes and multiple longitudes
    are provided then they are paired element-wise.

    If a cell contains more than one of the latitude-longitude
    locations, then by default that cell's index appears only once in
    the output (see the *unique* parameter).

    If a latitude-longitude location lies outside of all cells then by
    default it is ignored (see the *unique* parameter).

    This is a worker function that is called by `cf.locate`.

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.locate`

    :Parameters:

        y: array_like or `None`
            The Y coordinates of the points to locate, either in
            degrees north or in the same units as the 1-d Y
            coordinates that define cells of the discrete axis. A
            scalar or multi-dimensional array is flattened to 1-d. If
            *lat* has size 1 and *lon* has size greater than 1 then
            *lat* is automatically broadcast to the shape of
            *lon*. `None` is equivalent to an empty list.
        
        x: array_like or `None`
            The X coordinates of the points to locate, either in
            degrees east or in the same units as the 1-d X coordinates
            that define cells of the discrete axis. A scalar or
            multi-dimensional array is flattened to 1-d. If *lat* has
            size 1 and *lon* has size greater than 1 then *lat* is
            automatically broadcast to the shape of *lon*. `None` is
            equivalent to an empty list.

        f: `Field` or `Domain`
            The field or domain containing the UGRID grid.

        latlon: `bool`, optional
            If True (the default) then *y* is assumed to be in units
            of degrees north, and *x* in units of degees east. If
            False then *y* and *x* are assumed to be in units of the
            node coordinates in the domain.

            When the domain has a non-latitude_longitude
            coordinate reference system (CRS), it is still allowed
            to provide *y* and *x* points in latitude and
            longitude (i.e. *latlon* is True). In this case, the
            latitude and longitude points will internally be
            automatically converted to their equivalent points in
            the CRS of the domain.

        unique: `bool`, optional
            If True (the default) then return sorted cell indices
            without duplicates, nor placeholders for locations are
            out-of-domain (i.e that do not lie inside any cell). If
            False then an index is returned for every location,
            including placeholders for out-of-domain points.

            The out-of-domain placeholder is ``-N``, where ``N`` is
            the size of the discrete axis, which is always a non-valid
            index.

        return_operator: `bool`, optional
            If True then do not locate cells, but return an operator
            that is able to do the locating. The operator is a
            function that takes the same *y*, *x* and *unique*
            arguments as this function, and may also have extra
            configuration parameters. When multiple `cf.locate` calls
            for the same field or domain are expected, using the
            operator to do the locating can be much faster.

            When returning the operator, the *y* and *x* parameters
            are ignored (so may both be set to `None`), and the
            *unique* parameter is also ignored.

            For UGRID domains the performance benefits can be large,
            because the kd-tree used for nearest-neighbour lookups is
            cached inside the operator, and so does not need to be
            recreated each time a set of points is located. For
            instance, in one test with a UGRID domain with ~5 million
            faces, making the kd-tree during the creation of the
            operator takes ~20 seconds, but using the operator's
            cached kd-tree to find the cell index for each point
            location takes only 50 microseconds.

    :Returns:

        `numpy.ndarray` or function
            1-d cell indices for the discrete axis that contain the
            (Y, X) locations. See the *unique* parameter.

            If *f* is `None` then a function is returned that
            calculates cell indices for a given field or domain.

            If *return_operator* is True then a function is returned
            that, for the field or domain defined by *f*, calculates
            cell indices for given (Y, X) locations.

    """
    # Get the face_node_connectivity array, and normalise it.
    key, face_node_connectivity = f.domain_topology(
        filter_by_cell=("face",), item=True
    )
    face_node_connectivity = face_node_connectivity.normalise(
        start_index=0, remove_empty_columns=True
    )

    nodes = f.node_coordinates()
    if nodes[0].Y and nodes[1].X:
        nodes_y, nodes_x = nodes
    elif nodes[0].X and nodes[1].Y:
        nodes_x, nodes_y = nodes
    else:
        raise ValueError(
            "Can't locate points for a UGRID domain that does not have "
            f"well-defined X and Y node coordinates: {f!r}"
        )

    # Find the coordinate references for the UGRID face cell
    # coordinates (if there are any)
    cr = None
    cr_latlon = None
    coordinate_references = f.coordinate_references(todict=True)
    if coordinate_references:
        # Find the UGRID face coordinates
        ugrid_axis = f.get_data_axes(key)[0]
        coordinates = f.auxiliary_coordinates(
            filter_by_axis=(ugrid_axis,), axis_mode="exact", todict=True
        )
        coordinate_keys = [k for k, c in coordinates.items() if c.X or c.Y]
        if len(coordinate_keys) != 2:
            raise ValueError("TODO")

        # Find the coordinate references
        for c in coordinate_references.values():
            if c.coordinates().issuperset(coordinate_keys):
                if cr is None:
                    cr = c
                else:
                    raise ValueError("TODO")
            elif cr.identity("") == "grid_mapping_name:latitude_longitude":
                cr_latlon = c

    # Initialise a polygon locator instance
    locator = PolygonLocator(
        nodes_y.array,
        nodes_x.array,
        face_node_connectivity.array,
        cr=cr,
        cr_latlon=cr_latlon,
        atol=cf_atol(),
        units_y=nodes_y.Units,
        units_x=nodes_x.Units,
    )

    if return_operator:
        return locator.locate

    # Stll here? The find the indices for the points
    index = locator.locate(y, x, latlon=latlon, unique=unique)
    return index


class PolygonLocator:
    """Point-in-mesh locator for UGRID unstructured meshes.

    Provides O(log N) spatial indexing and bulk point-in-mesh location
    for arbitrary planar or spherical unstructured meshes, supporting
    arbitrary N-gons (triangles, quads, pentagons, hexagons) with both
    convex and concave geometries.

    Decomposes polygon faces into valid interior sub-triangles using a
    hybrid pipeline: vectorized fan-triangulation for convex faces and
    an ear-clipping fallback algorithm for concave elements.

    Point location queries execute a two-stage filter-and-verify
    algorithm:

    1. **Spatial Filtering (O(log N)):** Employs a
       `scipy.spatial.cKDTree` built on sub-triangle centroids to
       fetch candidate faces. Out-of-domain query points are rejected
       in O(1) time via a precomputed maximum cell radius bound.

    2. **Exact Verification (O(1)):** Evaluates a vectorised scalar
       triple product point-in-triangle test using `atol` for exact
       edge and boundary resolution.

    :Notes:

    - **N-vector geometry:** Spherical coordinates are represented as
      3-d N-vectors. Vector operations replace spherical trigonometry,
      rendering geometric tests invariant to coordinate branch cuts or
      poles.

    - **Vectorised fan triangulation:** Arbitrary ``n``-agons are
      decomposed into ``n-2`` sub-triangles in a single vectorised
      pass across all faces simultaneously.

    - **Progressive search guarantee:** If a candidate search with
      initial `k_candidates` fails to locate an interior point in
      complex boundary regions, the search depth scales progressively
      (multiplying `k_candidates` by 4 each iteration, bounded by
      `max_cell_radius`), guaranteeing 100% location reliability
      without degrading performance for standard points.

    :Attributes:

    `atol`: Absolute tolerance for floating point comparisons.

    `k_candidates`: The initial value for number of nearest triangle
    centroids retrieved from the KD-Tree spatial index for candidate
    evaluation, derived from maximum node valence (i.e. tha maximum
    number of sub-triangles that meet at a single node in a mesh) and
    99th-percentile cell aspect ratios.

    `max_cell_radius`: Maximum 3-d Euclidean distance from any
    sub-triangle centroid to one its vertices across the entire
    mesh. Defines the physical threshold for O(1) out-of-bounds
    rejection.

    `max_nodes`: The maximum number of nodes for any face cell.

    `n_faces`: The number of face cells in the mesh.

    `nodes`: Array of shape ``(n_nodes, 3)`` or ``(n_nodes,
    2)``. Either 3-d unit N-vectors representing all unique mesh nodes
    on the unit sphere; or the 2-d Cartesian coordinates of each node.

    `parent_face_ids`: Array of shape ``(n_sub_triangles,)``. Maps
    each sub-triangle back to its 0-based parent face index in the
    original `face_nodes` matrix.

    `tree`: `scipy.spatial.cKDTree` 3-d KD-Tree spatial index
    constructed over normalised sub-triangle centroids.

    `triangle_nodes`: Array of int Array of shape ``(n_sub_triangles,
    3)``. Node indices defining all sub-triangles generated by
    fan-triangulation of polygon faces.

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.locate`

    """

    def __init__(
        self,
        nodes_y,
        nodes_x,
        face_nodes,
        cr=None,
        cr_latlon=None,
        atol=1e-12,
        units_y=None,
        units_x=None,
        _test=False,
    ):
        """**Initialisation**

        :Parameters:

            nodes_y: 1-d array-like
                Laatitudes or northing for all unique mesh nodes.

            nodes_x: 1-d array-like
                Longitudes or eastings for all unique mesh nodes.

            face_nodes : 2-d array-like
                Integer connectivity matrix of shape ``(n_faces,
                max_nodes_per_face)``.

            cr: `None` or `CoordinateReference`, optional
                 TODO

            cr_latlon: `None` or `CoordinateReference`, optional
                 TODO

            atol: `float`, optional
                Absolute tolerance for floating point comparisons.

            units_y: `Units`, optional
                TODO

            units_x: `Units`, optional
                TODO

            _test: `bool`, optional
                TODO

        """
        from scipy.spatial import cKDTree

        self.cr = cr
        self.cr_latlon = cr_latlon
        self.atol = float(atol)
        self.units_y = units_y
        self.units_x = units_x
        self._test = _test

        # Find the grid mapping name
        if cr is None:
            self.grid_mapping = "latitude_longitude"
        elif _test and isinstance(cr, str):
            self.grid_mapping = cr
        else:
            identity = cr.identity("")
            if identity.startswith("grid_mapping_name:"):
                self.grid_mapping = identity[18:]
            else:
                raise ValueError("TODO")

        self.spherical = self.grid_mapping in (
            "latitude_longitude",
            "rotated_latitude_longitude",
        )

        # Build node coordinates array
        if self.spherical:
            # Spherical polar node coordinates (possibly rotated):
            # Convert to 3-d N-vectors. Array shape is (n_nodes, 3).
            nodes_x = np.radians(nodes_x)
            nodes_y = np.radians(nodes_y)
            self.nodes = np.column_stack(
                [
                    np.cos(nodes_y) * np.cos(nodes_x),
                    np.cos(nodes_y) * np.sin(nodes_x),
                    np.sin(nodes_y),
                ]
            )
        else:
            # 2-d Cartesian node coordinates. Array shape is (n_nodes,
            # 2).
            self.nodes = np.column_stack([nodes_x, nodes_y])

        del nodes_x, nodes_y

        # Get the face_node_connectivity mask (True where a face has
        # fewer than the maximum number of nords per face)
        mask = np.ma.getmaskarray(face_nodes)
        if np.ma.isMA(face_nodes):
            face_nodes = np.ma.getdata(face_nodes)
        else:
            face_nodes = np.asanyarray(face_nodes)

        self.n_faces, self.max_nodes = face_nodes.shape

        # Vectorized convexity check across all faces simultaneously
        is_convex = self._check_convexity(face_nodes, mask)

        triangle_list = []
        parent_list = []
        parents_base = np.arange(self.n_faces, dtype=int)

        # Triangulation of convex faces using fan trinagulation
        # (https://en.wikipedia.org/wiki/Fan_triangulation)
        convex_faces = face_nodes[is_convex]
        convex_mask = mask[is_convex]
        convex_parents = parents_base[is_convex]

        for k in range(1, self.max_nodes - 1):
            v0 = convex_faces[:, 0]
            vk = convex_faces[:, k]
            vk1 = convex_faces[:, k + 1]

            valid_k = (
                (~convex_mask[:, 0])
                & (~convex_mask[:, k])
                & (~convex_mask[:, k + 1])
            )

            tri_k = np.column_stack([v0, vk, vk1])[valid_k]
            parents_k = convex_parents[valid_k]

            triangle_list.append(tri_k)
            parent_list.append(parents_k)

        del convex_faces, convex_mask, convex_parents

        # Triangulation of concave faces using the ear clipping
        # algorithm
        # (https://en.wikipedia.org/wiki/Polygon_triangulation)
        non_convex_indices = np.where(~is_convex)[0]
        del is_convex

        local_tris = None
        local_tri_nodes = None
        for face_idx in non_convex_indices:
            row_mask = mask[face_idx]
            valid_node_ids = face_nodes[face_idx, ~row_mask]
            if len(valid_node_ids) < 3:
                continue

            face_verts = self.nodes[valid_node_ids]
            local_tris = self._ear_clip(face_verts)

            local_tri_nodes = [
                [
                    valid_node_ids[t[0]],
                    valid_node_ids[t[1]],
                    valid_node_ids[t[2]],
                ]
                for t in local_tris
            ]
            if local_tri_nodes:
                triangle_list.append(np.array(local_tri_nodes, dtype=int))
                parent_list.append(
                    np.full(len(local_tris), face_idx, dtype=int)
                )

        del mask, local_tri_nodes, local_tris, non_convex_indices

        # Build sub-triangle data structures and cKDTree spatial index
        if triangle_list:
            triangle_nodes = np.vstack(triangle_list)
        else:
            triangle_nodes = np.empty((0, 3), dtype=int)

        del triangle_list

        if len(triangle_nodes) > 0:
            self.triangle_nodes = np.vstack(triangle_nodes)
            del triangle_nodes

            self.parent_face_ids = np.concatenate(parent_list)
            del parent_list

            triangle_vertices = self.nodes[self.triangle_nodes]
            centroids = triangle_vertices.mean(axis=1)
            if self.spherical:
                centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

            self.tree = cKDTree(centroids)

            # Compute the maximum cell radius across the entire mesh =
            # the maximum distance over all triangles from the
            # triangle centroid to any one of its three vertices. If a
            # point is greater than this distance away from all
            # triangle centroids, then the point must lie completely
            # outside of the mesh (and therefore not lie in any grid
            # cell).
            #
            # d_v = distances from vertices to normalised sub-triangle
            #       centroid
            d_v = np.linalg.norm(
                triangle_vertices - centroids[:, None, :], axis=2
            )
            del centroids, triangle_vertices

            self.max_cell_radius = float(d_v.max())

            # Infer an initial value for k_candidates = The number of
            # nearest triangle centroids retrieved from the KD-Tree
            # spatial index for candidate evaluation.
            #
            # k_candidates has to be less or equal to the total number
            # of triangles. The initial value and should be small
            # enough to be sufficient for most points. For a given
            # point, if a triangle (of a face) cannot be found to
            # contain it, k_candidates will be increased in the
            # `locate` method until we find one that does (if
            # approriate).
            #
            # The initial value is derived from maximum node valence
            # (i.e. tha maximum number of sub-triangles that meet at a
            # single node in a mesh) and 99th-percentile cell aspect
            # ratios.
            node_counts = np.bincount(self.triangle_nodes.reshape(-1))
            max_degree = node_counts.max()
            del node_counts

            min_d = np.maximum(d_v.min(axis=1), self.atol)
            aspect_multiplier = float(
                np.percentile(d_v.max(axis=1) / min_d, 99)
            )
            del d_v, min_d

            inferred_k = int(np.ceil(2 * max_degree * aspect_multiplier))
            self.k_candidates = max(
                8, min(inferred_k, len(self.triangle_nodes))
            )
        else:
            self.triangle_nodes = np.empty((0, 3), dtype=int)
            self.parent_face_ids = np.empty((0,), dtype=int)
            self.k_candidates = 0
            self.max_cell_radius = 0.0
            self.tree = None

    def _check_convexity(self, face_nodes, mask):
        """Vectorized convexity check across all faces."""
        if self.max_nodes <= 3:
            return np.ones(self.n_faces, dtype=bool)

        n_verts = (~mask).sum(axis=1)

        # Note: Map masked slots to dummy node 0 for safe vectorized
        # slicing; invalid entries are ignored later on.
        safe_indices = np.where(mask, 0, face_nodes)

        verts = self.nodes[safe_indices]  # (n_faces, max_nodes, D)

        col_idx = np.arange(self.max_nodes)
        row_idx = np.arange(self.n_faces)[:, None]

        next_col = np.where(
            col_idx[None, :] < n_verts[:, None],
            (col_idx[None, :] + 1) % n_verts[:, None],
            col_idx[None, :],
        )
        next_next_col = np.where(
            col_idx[None, :] < n_verts[:, None],
            (col_idx[None, :] + 2) % n_verts[:, None],
            col_idx[None, :],
        )

        v_curr = verts
        v_next = verts[row_idx, next_col]
        v_next_next = verts[row_idx, next_next_col]

        edges1 = v_next - v_curr
        edges2 = v_next_next - v_next

        valid_cross_mask = col_idx[None, :] < n_verts[:, None]

        if self.spherical:
            valid_3d = valid_cross_mask[:, :, None]

            centroids = (verts * valid_3d).sum(axis=1) / n_verts[:, None]
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

            crosses_3d = np.cross(edges1, edges2)
            dots = (crosses_3d * centroids[:, None, :]).sum(axis=2)
            del crosses_3d, centroids

            is_pos = np.all((dots > self.atol) | ~valid_cross_mask, axis=1)
            is_neg = np.all((dots < -self.atol) | ~valid_cross_mask, axis=1)
            del dots
        else:
            crosses = (
                edges1[:, :, 0] * edges2[:, :, 1]
                - edges1[:, :, 1] * edges2[:, :, 0]
            )

            is_pos = np.all((crosses > self.atol) | ~valid_cross_mask, axis=1)
            is_neg = np.all((crosses < -self.atol) | ~valid_cross_mask, axis=1)
            del crosses

        del valid_cross_mask

        non_zero = is_pos | is_neg
        del is_pos, is_neg

        return non_zero & (n_verts >= 3)

    def _convert(self, lat, lon):
        """Convert lat/lon points to y/x points in a different CRS.

        The new CRS is defined by `self.cr`. The CRS for the latlon
        points is defined by `self.cr_latlon`. If `self.cr_latlon` is
        `None` then the shape of the ellipsoid is taken from
        `self.cr`.

        The y and x points will have the same units as the Y node and
        X node corodinates, respectively.

        .. versionadded:: NEXTVERSION

        :Parameters:

            lat: `numpy.ndarray`
                The latitudes, in a 1-d array.

            lon: `numpy.ndarray`
                The longitudes, in a 1-d array.

        :Returns:

            `numpy.ndarray`, `numpy.ndarray`
                The 1-d y and x arrays.

        """
        try:
            import pyproj
        except Exception:
            raise ImportError(
                "Can't locate points: Must install the 'pyproj' library"
            )

        from .mixin.utils.grid_mapping import create_projection_CRS
        from .units import Units

        # Create the source latitude-longitude CRS
        cr_latlon = self.cr_latlon
        if cr_latlon is None:
            cr_latlon = self.cr

        proj_latlon = create_projection_CRS(cr_latlon, "latitude_longitude")
        if proj_latlon is None:
            raise ValueError(
                "Can't locate points: Unable to create a "
                f"latitude_longitude pyproj.CRS object from {cr_latlon!r}"
            )

        # Create the destination CRS
        proj_dst = create_projection_CRS(self.cr, self.grid_mapping)
        if proj_dst is None:
            raise ValueError(
                "Can't locate points: Unable to create a "
                f"{self.grid_mapping} pyproj.CRS object for {self.cr!r}"
            )

        # Create the transform function that converts latitude and
        # longitudes to y and x coordinates
        try:
            transformer = pyproj.Transformer.from_crs(
                proj_latlon, proj_dst, always_xy=True
            )
        except Exception as error:
            raise ValueError(
                "Can't locate points: Unable to create a "
                "pyproj.transformer.Transformer object for converting "
                f"lat/lon points to y/x: {error}"
            )

        # Convert the latitudes and longitudes to y and x coordinates
        try:
            x, y = transformer.transform(
                lon, lat, errcheck=True, radians=False
            )
        except Exception as error:
            raise ValueError(
                "Can't locate points: Unable to convert lat/lon points "
                f"to y/x: {error}"
            )

        # Make sure that y and x have the same units as the node
        # coordinates
        metres = Units("m")
        if self.units_x.equivalent(metres):
            if self.units_x != metres:
                try:
                    Units.conform(x, metres, self.units_x, inplace=True)
                except ValueError:
                    raise ValueError(
                        "Can't locate points: Unable to convert longitude "
                        "points to x points in units of metres"
                    )

            if self.units_y != metres:
                try:
                    Units.conform(y, metres, self.units_y, inplace=True)
                except ValueError:
                    raise ValueError(
                        "Can't locate points: Unable to convert latitude "
                        "points to y points in units of metres"
                    )

        return y, x

    def _ear_clip(self, polygon):
        """Ear clipping algorithm for individual concave polygons."""
        n = len(polygon)
        indices = list(range(n))
        triangles = []

        if self.spherical:
            c = polygon.mean(axis=0)
            c /= np.linalg.norm(c)

            def anticlockwise(v0, v1, v2):
                """True if vertices are arranged anticlockwise."""
                return np.dot(np.cross(v1 - v0, v2 - v1), c) > 1e-12

            def point_in_triangle(pt, v0, v1, v2):
                """True if point pt is inside triangle (v0, v1, v2)."""
                v_arr = np.array([v0, v1, v2])
                crosses = np.cross(v_arr, np.roll(v_arr, -1, axis=0))
                return np.all(np.dot(crosses, pt) >= -self.atol)

        else:

            def anticlockwise(v0, v1, v2):
                """True if vertices are arranged anticlockwise."""
                return (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (
                    v2[0] - v0[0]
                ) > 1e-12

            def point_in_triangle(pt, v0, v1, v2):
                """True if point pt is inside triangle (v0, v1, v2)."""
                v_arr = np.array([v0, v1, v2])
                v_next = np.roll(v_arr, -1, axis=0)
                crosses = (v_next[:, 0] - v_arr[:, 0]) * (
                    pt[1] - v_arr[:, 1]
                ) - (v_next[:, 1] - v_arr[:, 1]) * (pt[0] - v_arr[:, 0])
                return np.all(crosses >= -self.atol)

        count = 0
        while len(indices) > 3 and count < 2 * len(indices):
            ear_found = False
            n_curr = len(indices)
            for i in range(n_curr):
                prev_i = indices[(i - 1) % n_curr]
                curr_i = indices[i]
                next_i = indices[(i + 1) % n_curr]

                p_prev, p_curr, p_next = (
                    polygon[prev_i],
                    polygon[curr_i],
                    polygon[next_i],
                )

                if not anticlockwise(p_prev, p_curr, p_next):
                    continue

                contains_other = False
                for j in range(n_curr):
                    other_i = indices[j]
                    if other_i in (prev_i, curr_i, next_i):
                        continue

                    if point_in_triangle(
                        polygon[other_i], p_prev, p_curr, p_next
                    ):
                        contains_other = True
                        break

                if not contains_other:
                    triangles.append((prev_i, curr_i, next_i))
                    indices.pop(i)
                    ear_found = True
                    break

            if not ear_found:
                break

            count += 1

        if len(indices) == 3:
            triangles.append((indices[0], indices[1], indices[2]))

        return triangles

    def locate(self, y, x, latlon=True, unique=True, k_candidates=None):
        """Find the indices of UGRID face cells containing points.

        :Method:

        Performs a spatial lookup using a two-stage filter-and-verify
        algorithm: O(log N) nearest-centroid retrieval via the cached
        KD-Tree, followed by an O(1) vectorised scalar triple product
        check.

        - **Fast out-of-mesh rejection**: A point whose distance to
          its nearest centroid exceeds the pre-determined maximum cell
          radius is flagged as outside the mesh instantly without,
          evaluating vector cross-products.

        - **Progressive fallback guarantee:**: If a point is within
          the pre-determined maximum cell radius of the mesh but fails
          initial verification, `locate` dynamically expands the
          candidate search radius until the point is either
          successfully located or it is shown to be outside of the
          mesh.

        .. versionadded:: NEXTVERSION

        .. seealso:: `cf.locate`

        :Parameters:

            y: array_like or `None`
                The Y coordinates of the points to locate, either in
                degrees north or in the same units as the 1-d Y
                coordinates that define cells of the discrete axis. A
                scalar or multi-dimensional array is flattened to
                1-d. If *lat* has size 1 and *lon* has size greater
                than 1 then *lat* is automatically broadcast to the
                shape of *lon*. `None` is equivalent to an empty list.
            
            x: array_like or `None`
                The X coordinates of the points to locate, either in
                degrees east or in the same units as the 1-d X
                coordinates that define cells of the discrete axis. A
                scalar or multi-dimensional array is flattened to
                1-d. If *lat* has size 1 and *lon* has size greater
                than 1 then *lat* is automatically broadcast to the
                shape of *lon*. `None` is equivalent to an empty list.

            latlon: `bool`, optional
                If True (the default) then *y* is assumed to be in
                units of degrees north, and *x* in units of degees
                east. If False then *y* and *x* are assumed to be in
                units of the node coordinates in the domain.

                When the domain has a non-latitude_longitude
                coordinate reference system (CRS), it is still allowed
                to provide *y* and *x* points in latitude and
                longitude (i.e. *latlon* is True). In this case, the
                latitude and longitude points will internally be
                automatically converted to their equivalent points in
                the CRS of the domain.

            unique: `bool`, optional
                If True (the default) then return sorted cell indices
                without duplicates, nor placeholders for locations are
                out-of-domain (i.e that do not lie inside any
                cell). If False then an index is returned for every
                location, including placeholders for out-of-domain
                points.

                The out-of-domain placeholder is ``-N``, where ``N``
                is the size of the discrete axis, which is always a
                non-valid index.

            k_candidates: `int` or `None`, optional

        :Returns:

            `numpy.ndarray`
                1-d cell indices for the discrete axis that contain the
                (Y, X) locations. See the *unique* parameter.

        """
        # Make sure that 'y' and 'x' are 1-d numpy arrays with the
        # same number of elements
        if y is None:
            y = []

        if x is None:
            x = []

        y = np.atleast_1d(y).reshape(-1)
        x = np.atleast_1d(x).reshape(-1)
        if y.size == 1 and x.size > 1:
            y = np.full_like(x, y, dtype=y.dtype)
        elif x.size == 1 and y.size > 1:
            x = np.full_like(y, x, dtype=x.dtype)
        elif y.size != x.size:
            raise ValueError(
                "Incompatible y and x sizes in "
                f"{self.__class__.__name__}.locate: {y.size} != {x.size}"
            )

        if not y.size or not x.size:
            # Return an empty index list if there are no x or y coordinates
            return np.array([], dtype=int)

        n_points = y.size
        results = np.full(n_points, -1, dtype=int)

        n_triangles = len(self.triangle_nodes)
        if n_triangles == 0:
            return results

        if self.grid_mapping == "latitude_longitude":
            if not latlon:
                # Nodes are defined in latitude_longitude, but the
                # points are not.
                raise ValueError("TODO")
        elif latlon:
            # Points are defined in latitude_longitude, but the nodes
            # are not, so convert the points to the CRS of the nodes.
            y, x = self._convert(y, x)

        if self.spherical:
            # Spherical polar point coordinates (possibly rotated):
            # Convert to 3-d N-vectors. Array shape is (n_nodes, 3).
            if np.abs(y).max() > 90:
                raise ValueError(
                    "Can't find cell locations: Latitudes must be in "
                    f"the range [-90, 90]. Got: {y}"
                )

            # Convert spherical polar coordinates to 3-d N-vectors
            # (unit sphere). Array shape is (n_points, 3).
            y = np.radians(y)
            x = np.radians(x)
            Q = np.column_stack(
                [
                    np.cos(y) * np.cos(x),
                    np.cos(y) * np.sin(x),
                    np.sin(y),
                ]
            )
        else:
            # 2-d Cartesian point coordinates. Array shape is
            # (n_nodes, 2).
            Q = np.column_stack([x, y])

        del y, x

        unmatched_indices = np.arange(n_points)
        max_distance_bound = self.max_cell_radius + self.atol

        k_current = self.k_candidates if k_candidates is None else k_candidates
        k_current = min(k_current, n_triangles)

        while len(unmatched_indices) > 0:
            Q_unmatched = Q[unmatched_indices]

            distances, triangle_indices = self.tree.query(
                Q_unmatched, k=k_current
            )

            if k_current == 1:
                distances = distances[:, None]
                triangle_indices = triangle_indices[:, None]

            newly_matched = []
            outside_mesh = []

            for sub_idx, point_idx in enumerate(unmatched_indices):
                q_vec = Q[point_idx]
                pt_dists = distances[sub_idx]
                candidate_triangles = triangle_indices[sub_idx]

                if pt_dists[0] > max_distance_bound:
                    outside_mesh.append(point_idx)
                    continue

                valid_mask = (pt_dists <= max_distance_bound) & (
                    candidate_triangles < n_triangles
                )
                valid_triangles = candidate_triangles[valid_mask]

                for tri_idx in valid_triangles:
                    v = self.nodes[self.triangle_nodes[tri_idx]]

                    if self.spherical:
                        crosses = np.cross(v, np.roll(v, -1, axis=0))
                        is_inside = np.all(
                            np.dot(crosses, q_vec) >= -self.atol
                        )
                    else:
                        v_next = np.roll(v, -1, axis=0)
                        crosses = (v_next[:, 0] - v[:, 0]) * (
                            q_vec[1] - v[:, 1]
                        ) - (v_next[:, 1] - v[:, 1]) * (q_vec[0] - v[:, 0])
                        is_inside = np.all(crosses >= -self.atol)

                    if is_inside:
                        results[point_idx] = self.parent_face_ids[tri_idx]
                        newly_matched.append(point_idx)
                        break

                if point_idx not in newly_matched and np.any(
                    pt_dists > max_distance_bound
                ):
                    outside_mesh.append(point_idx)

            resolved = newly_matched + outside_mesh
            if resolved:
                unmatched_indices = np.setdiff1d(
                    unmatched_indices, resolved, assume_unique=True
                )

            if k_current >= n_triangles or len(outside_mesh) == len(
                unmatched_indices
            ):
                break

            k_current = min(k_current * 4, n_triangles)

        size = results.size
        if unique and size:
            if size > 1:
                results = np.unique(results)

            if results[0] == -1:
                results = results[1:]
        elif -1 in results:
            # Replace indices of -1 (= outside of mesh) with an
            # invalid negative index
            results = np.where(results == -1, -self.n_faces, results)

        return results
