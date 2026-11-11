import numpy as np

from cf import atol as cf_atol


def _ugrid_locate(y, x, f, latlon=True, unique=True, return_operator=False):
    """Locate UGRID cells containing points.

    The cells must be defined by a horizontal UGRID discrete axis that
    has 1-d coordinates.

    The points are defined either within the domain's
    latitude_longitude coordinate system (*latlon* is True), or else
    within the coordinate system that defines the discrete axis cell
    coordinates (*latlon* is False).

    This is a worker function that is called by `cf.locate`.

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.locate`

    :Parameters:

        y: array_like or `None`

            The Y coordinates of the points to locate, either in units
            of degrees north within the domain's latitude_longitude
            coordinate system (``latlon=True``), or in the same units
            as the Y cell coordinates within the coordinate system of
            the discrete axis cell coordinates (``latlon=False``).

            A scalar or multi-dimensional array is flattened to
            1-d. If *y* has size 1 and *x* has size greater than 1
            then *y* is automatically broadcast to the shape of
            *x*. `None` is equivalent to an empty list.

        x: array_like or `None`

            The X coordinates of the points to locate, either in units
            of degrees east within the domain's latitude_longitude
            coordinate system (``latlon=True``), or in the same units
            as the X cell coordinates within the coordinate system of
            the discrete axis cell coordinates (``latlon=False``).

            A scalar or multi-dimensional array is flattened to
            1-d. If *x* has size 1 and *y* has size greater than 1
            then *x* is automatically broadcast to the shape of
            *y*. `None` is equivalent to an empty list.

        f: `Field` or `Domain`
            The field or domain containing the horizontal UGRID grid.

        latlon: `bool`, option

            If True (the default) then *y* and *x* are points in the
            latitude_longitude coordinate system that is explicitly or
            implicitly defined by the domain. If False then *y* and
            *x* are points in the coordinate system of the discrete
            axis cell coordinates.

            When the discrete axis cell coordinates have a
            latitude_longitude coordinate system, *latlon* must
            be True.

            When the discrete axis cell coordinates have a
            non-latitude_longitude coordinate system, it is still
            allowed to provide *y* and *x* points in latitude and
            longitude (by setting *latlon* to True). In this case, the
            latitude and longitude points will be automatically
            converted to equivalent points in the coordinate system of
            the discrete axis cell coordinates.

        unique: `bool`, optional

            If True (the default) then return sorted cell indices
            without duplicates, nor placeholders for points that are
            out-of-domain (i.e that do not lie inside any cell). If
            False then an index is returned for every point, including
            placeholders for out-of-domain points.

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
            The cell indices for the discrete axis that contain the
            points. See the *unique* parameter.

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
    # coordinates (if there are any). We might need them to onvert the
    # y and x points to a different corodiante system.
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
    """Point-in-polygon locator.

    Works for concave and convex cells (but not yet multi-part cells,
    or cells with holes).

    :Attributes:

    `atol`: Absolute tolerance for floating point comparisons.

    `n_sub_triangles`: The initial value for number of nearest triangle
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

            nodes_y: array-like
                The 1-d Y cooridnates for the unique mesh nodes.

            nodes_x: array-like
                The 1-d X cooridnates for the unique mesh nodes.

            face_nodes: array-like
                The 2-d integer face_node_connectivity matrix.

            cr: `None` or `CoordinateReference`, optional
                The coordinate system of the mesh coordinates. If
                `None` (the defaut), then a spherical
                latitude_longitude coordinate is assumed.

            cr_latlon: `None` or `CoordinateReference`, optional
                When points to be located are given by latitudes and
                longitudes, the latitude_longitude coordinate system
                of those points. If `None` (the defaut), then a
                spherical latitude_longitude coordinate is assumed.

            atol: `float`, optional
                The absolute tolerance used in the floating point
                comparisons that are required for identifying which
                cell a point might lie in.

            units_y: `Units`, optional
                TODO

            units_x: `Units`, optional
                TODO

            _test: `bool`, optional
                If True then all *cr* and *cr_latlon* to be strings in
                cases where the coordinate systems are not needed for
                point conversions. The sole purpose of this is to
                facilitate testing of the code, and in general this
                parameter should be set to False (the default).

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

        # The face_node_connectivity mask that has the same shape as
        # 'face_nodes' and is False for each valid vertex of each
        # face, and True where face_nodes is padded with -1.
        mask = np.ma.getmaskarray(face_nodes)

        if np.ma.isMA(face_nodes):
            # Fill missing values with -1. This is important, and gets
            # used in `_check_convexity`.
            face_nodes = face_nodes.filled(-1)
        else:
            face_nodes = np.asanyarray(face_nodes)

        self.n_faces, self.max_nodes = face_nodes.shape

        # Find out if each face is concave or convex
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
        non_convex_indices = np.where(~is_convex)[0].tolist()
        del is_convex

        for f_index in non_convex_indices:
            row_mask = mask[f_index]
            valid_node_ids = face_nodes[f_index, ~row_mask]
            if len(valid_node_ids) < 3:
                # Skip degenerate cells
                continue

            # Coordinates of each vertex of this cell
            face_vertices = self.nodes[valid_node_ids]

            # 2-d array of the 0-indexed relative vertex positions of
            # the sub-triangles within the polygon
            local_triangles = self._ear_clip(face_vertices)

            # The mesh node indices for each sub-triangle.
            local_triangle_nodes = valid_node_ids[local_triangles]

            if local_triangle_nodes.size:
                triangle_list.append(np.array(local_triangle_nodes, dtype=int))
                parent_list.append(
                    np.full(len(local_triangles), f_index, dtype=int)
                )

        del mask, non_convex_indices

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

            # Create the kd-tree used for nearest-triangle lookups
            self.tree = cKDTree(centroids)

            # Compute the maximum cell radius across the entire mesh =
            # the maximum distance over all triangles from the
            # triangle centroid to any one of its three vertices.#
            #
            # If a point is greater than this distance away from all
            # triangle centroids, then the point must lie completely
            # outside of the mesh (and therefore not lie in any grid
            # cell).
            self.max_cell_radius = (
                float(
                    np.linalg.norm(
                        triangle_vertices - centroids[:, None, :], axis=2
                    ).max()
                )
                + self.atol
            )

            del centroids, triangle_vertices
        else:
            self.triangle_nodes = np.empty((0, 3), dtype=int)
            self.parent_face_ids = np.empty((0,), dtype=int)
            self.max_cell_radius = 0.0
            self.tree = None

    def _check_convexity(self, face_nodes, mask):
        """Convexity check for all faces.

        .. versionadded:: NEXTVERSION

        :Parameters:

            face_nodes: `numpy.array`
                The face_node_connectivity array that contains the
                node index of each vertex of each cell, padded with
                ``-1`` when a cell has fewer than `max_nodes`
                vertices.

            mask: `numpy.array`
                The face_node_connectivity mask that has the same
                shape as *face_nodes* and is False for each valid
                vertex of each face, and True where *face_nodes* is
                padded with ``-1``.

        :Returns:

            `nump.ndarray`
                1-d Boolean array of size `n_faces` that is True or
                convex cells and False for concave cells.

        """
        if self.max_nodes <= 3:
            # All cells are triangles (or degenerate) => convex
            return np.ones(self.n_faces, dtype=bool)

        # Number of vertices for each cell
        n_vertices = (~mask).sum(axis=1)

        # Coordinates of each vertex of each cell.
        #
        # When a cell has fewer than 'self.max_nodes' vertices it is
        # padded with the coordinates of node index 'self.max_nodes -
        # 1' (by virtue of the fact that 'face_nodes' is padded with
        # -1), and such fake vertices are ignored later on via the
        # 'mask' boolean array.
        vertices = self.nodes[face_nodes]

        # Column indices
        columns = np.arange(self.max_nodes)

        # Row indices
        rows = np.arange(self.n_faces)[:, None]

        next_col = np.where(
            columns[None, :] < n_vertices[:, None],
            (columns[None, :] + 1) % n_vertices[:, None],
            columns[None, :],
        )
        next_next_col = np.where(
            columns[None, :] < n_vertices[:, None],
            (columns[None, :] + 2) % n_vertices[:, None],
            columns[None, :],
        )

        v_curr = vertices
        v_next = vertices[rows, next_col]
        v_next_next = vertices[rows, next_next_col]

        edges1 = v_next - v_curr
        del v_curr
        edges2 = v_next_next - v_next
        del v_next, v_next_next

        # Work out if each cell is convex or concave
        if self.spherical:
            valid_3d = ~mask[:, :, None]
            centroids = (vertices * valid_3d).sum(axis=1) / n_vertices[:, None]
            del valid_3d, vertices
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

            crosses_3d = np.cross(edges1, edges2)
            del edges1, edges2

            dots = (crosses_3d * centroids[:, None, :]).sum(axis=2)
            del crosses_3d, centroids

            is_pos = np.all((dots > self.atol) | mask, axis=1)
            is_neg = np.all((dots < -self.atol) | mask, axis=1)
            del dots
        else:
            del vertices

            crosses = (
                edges1[:, :, 0] * edges2[:, :, 1]
                - edges1[:, :, 1] * edges2[:, :, 0]
            )
            del edges1, edges2

            is_pos = np.all((crosses > self.atol) | mask, axis=1)
            is_neg = np.all((crosses < -self.atol) | mask, axis=1)
            del crosses

        non_zero = is_pos | is_neg
        del is_pos, is_neg

        return non_zero & (n_vertices >= 3)

    def _convert_from_latlon(self, lat, lon):
        """Convert lat/lon points to y/x points in a different CRS.

        The new CRS is defined by `self.cr`. The CRS for the latlon
        points is defined by `self.cr_latlon`. If `self.cr_latlon` is
        `None` then the shape of the ellipsoid is taken from
        `self.cr`.

        The returned y and x points will have the same units as the Y
        and X node corodinates, respectively.

        .. versionadded:: NEXTVERSION

        :Parameters:

            lat: `numpy.ndarray`
                The 1-d latitudes in degrees north.

            lon: `numpy.ndarray`
                The 1-d longitudes in degrees east.

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
                f"{self.grid_mapping} pyproj.CRS object from {self.cr!r}"
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
                f"latitude_longitude points to {self.grid_mapping} points: "
                f"{error}"
            )

        # Convert the latitudes and longitudes to y and x coordinates
        try:
            x, y = transformer.transform(
                lon, lat, errcheck=True, radians=False
            )
        except Exception as error:
            raise ValueError(
                "Can't locate points: Unable to convert latitude_longitude "
                f"to {self.grid_mapping} points using "
                f"pyproj.transformer.Transformer: {error}"
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
                        "Can't locate points: Unable to convert "
                        f"latitude_longitude points to {self.grid_mapping} "
                        f"points in units of metres"
                    )

        if self.units_y.equivalent(metres):
            if self.units_y != metres:
                try:
                    Units.conform(y, metres, self.units_y, inplace=True)
                except ValueError:
                    raise ValueError(
                        "Can't locate points: Unable to convert "
                        f"latitude_longitude points to {self.grid_mapping} "
                        f"points in units of metres"
                    )

        return y, x

    def _ear_clip(self, face_vertices):
        """Ear clipping algorithm for individual concave polygons.

        Decomposes the concave polygon into sub-triangles.

        .. versionadded:: NEXTVERSION

        :Parameters:

            face_vertices: `numpy.ndarray`
                Coordinates of each vertex of a polygon.

        :Returns:

            `numpy.array`
                2-d array of the 0-indexed relative vertex positions
                of the sub-triangles within the polygon.

        """
        n_vertices, dim = face_vertices.shape
        if n_vertices == 3:
            return np.array([[0, 1, 2]])

        # Define a cross product scalar helper, separtely for 3-d and
        # 2-d cases.
        if dim == 3:
            centroid = np.mean(face_vertices, axis=0)
            norm_val = np.linalg.norm(centroid)
            if norm_val > 0:
                normal = centroid / norm_val
            else:
                normal = np.array([0.0, 0.0, 1.0])

            def cross_scalar(v1, v2):
                return float(np.dot(np.cross(v1, v2), normal))

        else:

            def cross_scalar(v1, v2):
                # 2-d cross-product (determinant): v1_x * v2_y - v1_y
                # * v2_x (avoids numpy DeprecationWarning for 2-d
                # np.cross)
                return float(v1[0] * v2[1] - v1[1] * v2[0])

        # Determine winding order (anticlockwise vs. clockwise) from
        # total signed area
        total_area = 0.0
        for i in range(n_vertices):
            v1 = face_vertices[i]
            v2 = face_vertices[(i + 1) % n_vertices]
            total_area += cross_scalar(v1, v2)

        if total_area >= 0:
            sign = 1.0
        else:
            sign = -1.0

        indices = list(range(n_vertices))
        triangles = []

        def is_convex(a, b, c):
            v1 = face_vertices[b] - face_vertices[a]
            v2 = face_vertices[c] - face_vertices[a]
            return sign * cross_scalar(v1, v2) > self.atol

        def contains_no_other_points(a, b, c, idx_list):
            pa, pb, pc = face_vertices[a], face_vertices[b], face_vertices[c]
            for p_idx in idx_list:
                if p_idx in (a, b, c):
                    continue

                pt = face_vertices[p_idx]
                c1 = sign * cross_scalar(pb - pa, pt - pa)
                c2 = sign * cross_scalar(pc - pb, pt - pb)
                c3 = sign * cross_scalar(pa - pc, pt - pc)
                if c1 >= -self.atol and c2 >= -self.atol and c3 >= -self.atol:
                    return False

            return True

        # Ear clipping loop
        #
        # 'max_ops' doesn't alter execution under normal
        # conditions. It simply serves as defense-in-depth safety
        # against infinite loops if edge-case topology bugs or future
        # refactoring (such as to include geomerty cells) ever prevent
        # indices from shrinking.
        max_ops = n_vertices * n_vertices

        count = 0
        while len(indices) > 3 and count < max_ops:
            count += 1
            ear_found = False
            num_verts = len(indices)

            for i in range(num_verts):
                prev_i = indices[(i - 1) % num_verts]
                curr_i = indices[i]
                next_i = indices[(i + 1) % num_verts]

                if is_convex(prev_i, curr_i, next_i):
                    if contains_no_other_points(
                        prev_i, curr_i, next_i, indices
                    ):
                        triangles.append((prev_i, curr_i, next_i))
                        indices.pop(i)
                        ear_found = True
                        break

            # Safety fallback if floating-point edge cases prevent ear
            # detection
            if not ear_found:
                triangles.append(indices[:3])
                # Remove the middle vertex (the "tip" of the ear) of
                # the triangle
                indices.pop(1)

        if len(indices) == 3:
            triangles.append(indices)

        return np.array(triangles)

    def _ear_clip_old(self, polygon):
        """Ear clipping algorithm for individual concave polygons.

        :Parameters:

            polygon: `numpy.ndarray`
                Coordinates of each vertex of a polygon.

        :Returns:

            TODO

        """
        n = len(polygon)
        indices = list(range(n))
        triangles = []

        if self.spherical:
            c = polygon.mean(axis=0)
            c /= np.linalg.norm(c)

            def anticlockwise(v0, v1, v2):
                """True if vertices are arranged anticlockwise."""
                return np.dot(np.cross(v1 - v0, v2 - v1), c) > self.atol

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
                ) > self.atol

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

    def locate(self, y, x, latlon=True, unique=True, n_sub_triangles=8):
        """Locate UGRID cells containing points.

        The cells are defined by a horizontal UGRID discrete axis that
        has 1-d coordinates.

        The points are defined either within the domain's
        latitude_longitude coordinate system (*latlon* is True), or
        else within the coordinate system that defines the UGRID
        discrete axis cell coordinates (*latlon* is False).

        Works for concave and convex cells (but not yet multi-part
        cells, or cells with holes).

        :Method:

        Grid cells are internally decomposed into non-overlapping
        sub-triangles. For each target point, the *n_sub_triangles*
        sub-triangles whose centroids are closest to the point are
        initially retrieved and tested for containment. If no match is
        found, and the point has not been identified as being
        out-of-domain, then more sub-triangles with farther-away
        centroids are included, until a match is found or the point
        has been identified as not lying in any cell.

        .. versionadded:: NEXTVERSION

        .. seealso:: `cf.locate`

        :Parameters:

            y: array_like or `None`

                The Y coordinates of the points to locate, either in
                units of degrees north within the domain's
                latitude_longitude coordinate system
                (``latlon=True``), or in the same units as the Y cell
                coordinates within the coordinate system of the
                discrete axis cell coordinates (``latlon=False``).

                A scalar or multi-dimensional array is flattened to
                1-d. If *y* has size 1 and *x* has size greater than 1
                then *y* is automatically broadcast to the shape of
                *x*. `None` is equivalent to an empty list.

            x: array_like or `None`

                The X coordinates of the points to locate, either in
                units of degrees east within the domain's
                latitude_longitude coordinate system
                (``latlon=True``), or in the same units as the X cell
                coordinates within the coordinate system of the
                discrete axis cell coordinates (``latlon=False``).

                A scalar or multi-dimensional array is flattened to
                1-d. If *x* has size 1 and *y* has size greater than 1
                then *x* is automatically broadcast to the shape of
                *y*. `None` is equivalent to an empty list.

            latlon: `bool`, optional

                If True (the default) then *y* and *x* are points in
                the latitude_longitude coordinate system that is
                explicitly or implicitly defined by the domain. If
                False then *y* and *x* are points in the coordinate
                system of the discrete axis cell coordinates.

                When the discrete axis cell coordinates have a
                latitude_longitude coordinate system, *latlon* must be
                True.

                When the discrete axis cell coordinates have a
                non-latitude_longitude coordinate system, it is still
                allowed to provide *y* and *x* points in latitude and
                longitude (by setting *latlon* to True). In this case,
                the latitude and longitude points will be
                automatically converted to equivalent points in the
                coordinate system of the discrete axis cell
                coordinates.

            unique: `bool`, optional

                If True (the default) then return sorted cell indices
                without duplicates, nor placeholders for points that
                are out-of-domain (i.e that do not lie inside any
                cell). If False then an index is returned for every
                point, including placeholders for out-of-domain
                points.

                The out-of-domain placeholder is ``-N``, where ``N``
                is the size of the discrete axis, which is always a
                non-valid index.

            n_sub_triangles: `int`, optional

                The initial number of sub-triangles queried for each
                point, selected in order of nearest centroid distance
                to the point.

                Grid cells are internally decomposed into
                non-overlapping sub-triangles. For each target point,
                the *n_sub_triangles* sub-triangles whose centroids
                are closest to the point are initially retrieved and
                tested for containment. If no match is found, and the
                point has not been identified as being out-of-domain,
                then more sub-triangles with farther-away centroids
                are included, until a match is found or the point has
                been identified as not lying in any cell.

                The value of *n_sub_triangles* only affects the
                performance of the location operation, not the indices
                of the located cells. For any given mesh, there will
                be an optimal value that finds the requested points
                the fastest. By default *n_sub_triangles* is
                ``8``. This is usually sufficient for locating,
                without needing to widen the search radius, any point
                in a mesh composed of low aspect-ratio triangles,
                quadrilaterals, pectagons, or hexagons (because for
                such meshes, it would be unusual for more than eight
                sub-triangles to all share the same mesh node).

        :Returns:

            `numpy.ndarray`
                The cell indices for the discrete axis that contain
                the points. See the *unique* parameter.

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
                f"Incompatible y and x sizes: {y.size} != {x.size}"
            )

        if not y.size or not x.size:
            # Return an empty index list if there are no x or y coordinates
            return np.array([], dtype=int)

        # Initialise the out put array of indices. It is important
        # that it is initialised to -self.n_faces.
        n_points = y.size
        results = np.full(n_points, -self.n_faces, dtype=int)

        n_triangles = len(self.triangle_nodes)
        if n_triangles == 0:
            # There are only degenerate cells, so no points can lie in
            # them
            return results

        # Make sure that n_sub_triangles is >= 1 and (a requirement of
        # scipy.spatial.cKDTree) not greater than the total number of
        # triangles
        n_sub_triangles = int(n_sub_triangles)
        if n_sub_triangles < 1:
            raise ValueError(
                f"n_sub_triangles must be >= 1. Got: {n_sub_triangles}"
            )

        n_sub_triangles = min(n_sub_triangles, n_triangles)

        if self.grid_mapping == "latitude_longitude":
            if not latlon:
                # Nodes are defined in latitude_longitude, but the
                # points are not.
                raise ValueError(
                    "Must set latlon=True when the node coordinates are "
                    "defined by a latitude_longitude coordinate system"
                )

        elif latlon:
            # Points are defined in latitude_longitude, but the nodes
            # are not, so convert the points to the CRS of the nodes.
            y, x = self._convert_from_latlon(y, x)

        if self.spherical:
            # Spherical polar point coordinates (possibly rotated):
            # Convert to 3-d N-vectors. Array shape is (n_nodes, 3).
            if np.abs(y).max() > 90:
                raise ValueError(
                    "Can't find cell locations: Latitudes must be in "
                    f"the range [-90, 90]. Got: {y}"
                )

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
        max_cell_radius = self.max_cell_radius

        old_k = 0
        k_current = n_sub_triangles

        while len(unmatched_indices) > 0:
            Q_unmatched = Q[unmatched_indices]

            distances, triangle_indices = self.tree.query(
                Q_unmatched, k=k_current
            )

            if k_current == 1:
                distances = distances[:, None]
                triangle_indices = triangle_indices[:, None]

            if old_k:
                # No need to check the triangles we've already
                # rejected
                distances = distances[:, old_k:k_current]
                triangle_indices = triangle_indices[:, old_k:k_current]

            newly_matched = []
            outside_mesh = []

            for n, p_index in enumerate(unmatched_indices):
                # The coordinates of the point
                q_vec = Q[p_index]

                # The distance from the point for each triangle
                point_distances = distances[n]

                # The indices of the k_current closest triangles
                candidate_triangles = triangle_indices[n]

                if point_distances[0] > max_cell_radius:  # max_distance_bound:
                    # The distance of the closest triangle to the
                    # point is greater than maximum cell radius => the
                    # point must be outside the mesh.
                    outside_mesh.append(p_index)
                    continue

                # Get rid of triangles that distance of the triangle
                # to the point is greater than maximum cell radius
                # (the point definitely can't be in those).
                valid_mask = (point_distances <= max_cell_radius) & (
                    candidate_triangles < n_triangles
                )
                valid_triangles = candidate_triangles[valid_mask]

                # Perform a point-in-triangle test for each valid
                # triangle, checking the triangle closest to the point
                # first.
                for t_index in valid_triangles:
                    # Coordinates of this triangle's vertices
                    v = self.nodes[self.triangle_nodes[t_index]]

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
                        # The point is inside this triangle
                        results[p_index] = self.parent_face_ids[t_index]
                        newly_matched.append(p_index)
                        break

                if p_index not in newly_matched and np.any(
                    point_distances > max_cell_radius
                ):
                    outside_mesh.append(p_index)

            resolved = newly_matched + outside_mesh
            if resolved:
                unmatched_indices = np.setdiff1d(
                    unmatched_indices, resolved, assume_unique=True
                )

            if k_current >= n_triangles or len(outside_mesh) == len(
                unmatched_indices
            ):
                break

            # Still here? Then double the number of candidate
            # triangles for the next iteration
            old_k = k_current
            if k_current == 1:
                k_current = 2
            else:
                k_current = k_current * 2

            k_current = min(k_current, n_triangles)

        size = results.size
        if unique and size:
            # Return sorted cell indices without duplicates, nor
            # placeholders for points that are out-of-domain (i.e that
            # do not lie inside any cell).
            if size > 1:
                # Unique and sort
                results = np.unique(results)

            if results[0] < 0:
                # Remove out-of-domain placeholder
                results = results[1:]

        return results
