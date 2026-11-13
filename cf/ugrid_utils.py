import numpy as np
from cfdm import integer_dtype

from cf import atol as cf_atol


def _ugrid_locate(
    y, x, f, latlon=True, unique=True, atol=None, return_domain_operator=False
):
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
            of degrees north within the domain's (implied)
            latitude_longitude coordinate system (``latlon=True``), or
            in the same units as the Y cell coordinates within the
            coordinate system of the discrete axis cell coordinates
            (``latlon=False``).

            A scalar or multi-dimensional array is flattened to
            1-d. If *y* has size 1 and *x* has size greater than 1
            then *y* is automatically broadcast to the shape of
            *x*. `None` is equivalent to an empty list.

        x: array_like or `None`

            The X coordinates of the points to locate, either in units
            of degrees east within the domain's (implied)
            latitude_longitude coordinate system (``latlon=True``), or
            in the same units as the X cell coordinates within the
            coordinate system of the discrete axis cell coordinates
            (``latlon=False``).

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
            without duplicates, and without placeholders for points
            that are out-of-domain (i.e that do not lie inside any
            cell). If False then an index is returned for every point,
            including placeholders for out-of-domain points.

            The out-of-domain placeholder is ``-N``, where ``N`` is
            the size of the discrete axis, which is always a non-valid
            index.

        atol: `float`, optional

            The absolute tolerance used for geometric orientation and
            degeneracy tests during sub-triangle decomposition, and
            for point-in-triangle location checks. Specifically,
            *atol* resolves the sign (positive, negative, or zero) for
            small values of the cross-product (2-d planar) or scalar
            triple-product (3-d spherical) of two vectors that share a
            vertex. Small values arise when a sub-triangle is
            degenerate or has collinear edges, or when a point to
            locate lies on or very close to a sub-triangle edge. The
            default value is set by the `cf.atol` function. Note that all
            internal floating-point calculations are carried out at
            64-bit precision.

        return_domain_operator: `bool`, optional

            If True then do not locate cells, but return a callable
            operator that is able to do the locating for the domain
            defined by *f*. The operator is a function that takes the
            same *y*, *x*, *latlon*, and *unique* arguments as this
            function, and also the extra *k_initial* configuration
            parameter. When multiple `_ugrid_locate` calls for the
            same field or domain are expected, using the domain
            operator to do the locating can be much faster. In this
            case, the *y* and *x* parameters are ignored (so may both
            be set to `None`), and the *latlon* and *unique*
            parameters are also ignored.

            Using a domain operator can give a large increase in
            performance over repeated `_ugrid_locate` calls, because
            the kd-tree used to in the location of the points is
            cached inside the domain operator, and so does not need to
            be recreated each time a set of points is located. For
            instance, in one test with a UGRID domain with ~5 million
            faces, locating one point with `_ugrid_locate` takes ~20
            seconds, which is nearly all taken up with defining the
            sub-triangles and storing them in the kd-tree, and only 50
            microseconds is for locating the cell after that. In this
            case, creating the domain operator takes ~20 seconds (it
            needs to cache the kd-tree), but all subsequent uses of
            the operator are very fast.

    :Returns:

        `numpy.ndarray` or function
            The cell indices for the discrete axis that contain the
            points. See the *unique* parameter.

            If *f* is `None` then a function is returned that
            calculates cell indices for a given field or domain.

            If *return_domain_operator* is True then a function is
            returned that, for the field or domain defined by *f*,
            calculates cell indices for the given (Y, X) locations.

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
            raise ValueError(
                "Can't locate points: There are not both X and Y "
                f"face cell auxilliary coordinate constructs in {f!r}"
            )

        # Find the coordinate references
        for c in coordinate_references.values():
            if c.coordinates().issuperset(coordinate_keys):
                if cr is None:
                    cr = c
                else:
                    raise ValueError(
                        "Can't locate points: There are two coordinate "
                        "reference constucts for the X and Y face cell "
                        f"auxilliary coordinate constructs in {f!r}"
                    )
            elif cr.identity("") == "grid_mapping_name:latitude_longitude":
                cr_latlon = c

    # Initialise a polygon locator instance
    locator = PolygonWindow(
        nodes_y.array,
        nodes_x.array,
        face_node_connectivity.array,
        cr=cr,
        cr_latlon=cr_latlon,
        atol=cf_atol(),
        units_y=nodes_y.Units,
        units_x=nodes_x.Units,
    )

    if return_domain_operator:
        return locator.locate

    # Stll here? The find the indices for the points
    return locator.locate(y, x, latlon=latlon, unique=unique)


class PolygonWindow:
    """Locate polygon cells that contain points.

    Works for concave and convex cells (but not yet multi-part cells,
    or cells with holes).

    :Attributes:

    `atol`: Absolute tolerance for floating point comparisons.

    `k_initial`: The initial value for number of nearest triangle
    centroids retrieved from the KD-Tree spatial index for candidate
    evaluation, derived from maximum nTODOode valence (i.e. tha maximum
    number of sub-triangles that meet at a single node in a mesh) and
    99th-percentile cell aspect ratios.

    `max_cell_radius`: Maximum 3-d Euclidean distance from any
    sub-triangle centroid to one its vertices across the entire
    mesh. Defines the physical threshold for O(1) out-of-bounds
    rejection.

    `max_vertices`: The maximum number of vertices for any face cell.

    `n_cells`: The number of face cells in the mesh.

    `nodes`: Array of shape ``(n_nodes, 3)`` or ``(n_nodes,
    2)``. Either 3-d unit N-vectors representing all unique mesh nodes
    on the unit sphere; or the 2-d Cartesian coordinates of each node.

    `parent_face_ids`: Array of shape ``(n_triangles,)``. Maps each
    sub-triangle back to its 0-based parent face index in the original
    `face_nodes` matrix.

    `tree`: `scipy.spatial.cKDTree` 3-d KD-Tree spatial index
    constructed over normalised sub-triangle centroids.

    `triangle_nodes`: Array of int Array of shape ``(n_triangles,
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
        atol=None,
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

                The 2-d integer face_node_connectivity matrix. The
                node indices must lie in the range [0, N-1], where N
                is the number of unique nodes (i.e. the size of the
                *nodes_y* and *nodes_x* arrays). If some faces have
                fewer vertices than others then *face_nodes* must be a
                `numpy` masked array padded with masked elements.

            cr: `None` or `CoordinateReference`, optional

                The definition of the coordinate system of the mesh
                node coordinates given by *nodes_y* and *nodes_x*. If
                `None` (the defaut), then a spherical
                latitude_longitude coordinate system is assumed.

            cr_latlon: `None` or `CoordinateReference`, optional

                The definition of the latitude_longitude coordinate
                system when *latlon* is True. If `None` (the defaut),
                then the latitude_longitude coordinate system implied
                *cr* is used.

            atol: `float`, optional

                The absolute tolerance used for geometric orientation
                and degeneracy tests during sub-triangle
                decomposition, and for point-in-triangle location
                checks. Specifically, *atol* resolves the sign
                (positive, negative, or zero) for small values of the
                cross-product (2-d planar) or scalar triple-product
                (3-d spherical) of two vectors that share a
                vertex. Small values arise when a sub-triangle is
                degenerate or has collinear edges, or when a point to
                locate lies on or very close to a sub-triangle
                edge. By default, *atol* is ``1e-12``. Note that all
                internal floating-point calculations are carried out
                at 64-bit precision.

            units_y: `Units`or `None`, optional

                The units of the *nodes_y* coordinates. Only required
                if the points are defined within a latitude_longitude
                coordinate system (*latlon* is True) need to be
                converted to points within the coordinate system of
                the discrete axis cell coordinates.

            units_x: `Units` or `None`, optional

                The units of the *nodes_x* coordinates. Only required
                if the points are defined within a latitude_longitude
                coordinate system (*latlon* is True) need to be
                converted to points within the coordinate system of
                the discrete axis cell coordinates.

            _test: `bool`, optional
                If True then allow *cr* and *cr_latlon* to be strings
                in cases where the coordinate systems are not needed
                for point conversions. The sole purpose of this is to
                facilitate testing of the code, and in general this
                parameter should be set to False (the default).

        """
        from scipy.spatial import cKDTree

        self.cr = cr
        self.cr_latlon = cr_latlon
        self.units_y = units_y
        self.units_x = units_x
        self._test = _test

        # Parse atol
        atol = float(atol)
        if atol < 0:
            raise ValueError("atol must be non-negative")

        self.atol = atol

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
                raise ValueError(
                    "The cr parameter must be an appropriate "
                    f"cf.CoordinateReference object or None. Got: {cr!r}"
                )

        self.spherical = self.grid_mapping in (
            "latitude_longitude",
            "rotated_latitude_longitude",
        )

        # Build node coordinates array
        nodes_y = np.asarray(nodes_y, dtype=np.float64)
        nodes_x = np.asarray(nodes_x, dtype=np.float64)
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

        self.n_cells, self.max_vertices = face_nodes.shape

        # Find out if each face is concave or convex
        is_convex = self._check_convexity(face_nodes, mask, self.atol)

        triangle_list = []
        parent_list = []
        parents_base = np.arange(
            self.n_cells, dtype=integer_dtype(self.n_cells)
        )

        # Triangulation of convex polygons using fan triangulation
        # (https://en.wikipedia.org/wiki/Fan_triangulation)
        convex_faces = face_nodes[is_convex]
        convex_mask = mask[is_convex]
        convex_parents = parents_base[is_convex]

        k = 0
        vertex_0 = convex_faces[:, 0]
        vertex_k1 = convex_faces[:, k + 1]

        mask_0 = ~convex_mask[:, 0]
        mask_k1 = ~convex_mask[:, k + 1]

        for k in range(1, self.max_vertices - 1):
            vertex_k = vertex_k1
            vertex_k1 = convex_faces[:, k + 1]

            mask_k = mask_k1
            mask_k1 = ~convex_mask[:, k + 1]

            valid_k = mask_0 & mask_k & mask_k1

            triangles_k = np.column_stack([vertex_0, vertex_k, vertex_k1])[
                valid_k
            ]
            parents_k = convex_parents[valid_k]

            triangle_list.append(triangles_k)
            parent_list.append(parents_k)

        del (
            vertex_0,
            vertex_k,
            vertex_k1,
            mask_0,
            mask_k,
            mask_k1,
            triangles_k,
            parents_k,
            valid_k,
        )
        del convex_faces, convex_mask, convex_parents

        # Triangulation of concave polygons using the ear clipping
        # algorithm
        # (https://en.wikipedia.org/wiki/Polygon_triangulation)
        #
        # "Concave" here means any polygons that have not been
        # previously registered as being convex, and so it is possible
        # for a polygon that is actually convex to end up going
        # through the ear clipping algorithm.
        if is_convex.all():
            concave_indices = []
        else:
            concave_indices = np.where(~is_convex)[0]

        del is_convex

        for f_index in concave_indices:
            row_mask = mask[f_index]
            valid_node_ids = face_nodes[f_index, ~row_mask]
            if len(valid_node_ids) < 3:
                # Skip degenerate cells
                continue

            # Coordinates of each vertex of this cell
            face_vertices = self.nodes[valid_node_ids]

            # 2-d array of the 0-indexed relative vertex positions of
            # the sub-triangles within the polygon
            local_triangles = self._ear_clip(face_vertices, self.atol)

            # The mesh node indices for each sub-triangle
            local_triangle_nodes = valid_node_ids[local_triangles]

            if local_triangle_nodes.size:
                triangle_list.append(np.array(local_triangle_nodes, dtype=int))
                parent_list.append(
                    np.full(len(local_triangles), f_index, dtype=int)
                )

        del mask, concave_indices

        # Build sub-triangle data structures and cKDTree spatial index
        if triangle_list:
            triangle_nodes = np.vstack(triangle_list)
        else:
            triangle_nodes = np.empty((0, 3), dtype=int)

        del triangle_list

        if len(triangle_nodes):
            # The node indices for each sub-triangle, with shape
            # (n_cells, 3).
            #
            # E.g if there are 3 cells with 7 unique nodes decomposed
            # into a total of 5 sub-triangles, then triangle_nodes
            # could be [[2, 3, 1], [4, 5, 3], [6, 1, 3], [2, 1, 0],
            # [4, 3, 2]]
            self.triangle_nodes = np.vstack(triangle_nodes)
            del triangle_nodes

            # The cell index for each sub-triangle, which shape
            # (n_triangles,).
            #
            # E.g. if parent_face_ids is [0, 1, 2, 0, 1] then there
            # are 3 cells (0, 1, and 2). Cell 0 contains the two
            # sub-triangles with sub-triangle indices 0 and 3, cell 1
            # contains the two sub-triangles with sub-triangle indices
            # 1 and 4, and cell 2 contains the one sub-triangle with
            # sub-triangle index 2.
            self.parent_face_ids = np.concatenate(parent_list)
            del parent_list

            # The vertex coordinates for each sub-triangle, with shape
            # (n_triangles, 3, N) for N-d coordinates.
            triangle_vertices = self.nodes[self.triangle_nodes]

            # The centroid coordinates for each sub-triangle, with
            # shape (n_triangles, N) for N-d coordinates.
            centroids = triangle_vertices.mean(axis=1)
            if self.spherical:
                # The centroid calculated above lies in the sphere's
                # interior, so move it to the sphere's surface.
                centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

            # The kd-tree used for nearest-triangle lookups
            self.kdtree = cKDTree(centroids)

            # Compute the maximum cell radius across the entire mesh,
            # i.e. the maximum over all triangles of the straight-line
            # (Euclidean) distance from the triangle centroid to any
            # one of its three vertices.
            #
            # Note: In the spherical case, each straight-line chord
            # lies in the sphere's interior, and is not an arc on the
            # sphere's surface.
            #
            # If a point is greater than this Euclidean distance away
            # from all triangle centroids, then it is impossible for
            # it to lie in any sub-triangle, and therefore it can't
            # lie in any cell.
            #
            # We add the square root of atol because atol is defined
            # with implicit units of distance squared, and
            # max_cell_radius has units of just distance.
            self.max_cell_radius = (
                float(
                    np.linalg.norm(
                        triangle_vertices - centroids[:, None, :], axis=2
                    ).max()
                )
            ) + self.atol**0.5

            del centroids, triangle_vertices
        else:
            self.triangle_nodes = np.empty((0, 3), dtype=int)
            self.parent_face_ids = np.empty((0,), dtype=int)
            self.max_cell_radius = 0.0
            self.kdtree = None

    def _check_convexity(self, face_nodes, mask, atol):
        """Convexity check for all faces.

        :Method:

        Determines whether each cell is strictly convex by checking
        that all consecutive edge pairs turn in the same direction
        around the cell perimeter (all interior angles strictly less
        than 180 degrees).

        A cell is convex if all of its vertices "turn" consistently
        anticlockwise or consistently clockwise. The turn at a vertex
        is calculated as the cross product of the vertex's two edges
        (2-d case), or as the dot product of the cell centroid with
        the cross product of the vertex's two edges (3-d case). If
        turn > atol then it is anticlockwise and if turn < -atol then
        it is clockwise.

        If a cell has both anticlockwise and clockwise turns then the
        cell is concave.

        If -atol <= turn <= atol for any vertex, then the cell has
        collinear or degenerate edges, and is flagged as non-convex.

        .. versionadded:: NEXTVERSION

        :Parameters:

            face_nodes: `numpy.array`
                The face_node_connectivity array with shape (n_cells,
                max_vertices) that contains the node index of each
                vertex of each cell, padded with ``-1`` when a cell
                has fewer than `max_vertices` vertices.

            mask: `numpy.array`
                The face_node_connectivity mask that has the same
                shape as *face_nodes* and is False for each valid
                vertex of each face, and True where *face_nodes* is
                padded with ``-1`` for missing vertices.

            atol: `float`
                The postive absolute tolerance used to assess the turn
                of a vertex. See the method description.

        :Returns:

            `numpy.array`
                1-d Boolean array of shape (n_cells,) that is True for
                convex cells and False otherwise.

        """
        if self.max_vertices <= 3:
            # All cells are triangles (or degenerate)
            is_convex = (~mask).all(axis=1)
            return is_convex

        # Number of vertices for each cell
        n_vertices = (~mask).sum(axis=1)

        # Coordinates of cell's vertices, with shape (n_cells,
        # max_vertices, N) for N-d coordinates.
        #
        # When a cell has fewer than 'max_vertices' vertices, it is
        # padded with an arbitrary value, and such fake vertices are
        # ignored later on by applying the 'mask' Boolean array.
        vertices = self.nodes[face_nodes]

        # Column indices
        columns = np.arange(
            self.max_vertices, dtype=integer_dtype(self.max_vertices - 1)
        )
        columns = columns[None, :]

        # Row indices
        rows = np.arange(self.n_cells, dtype=integer_dtype(self.n_cells - 1))
        rows = rows[:, None]

        next_col = np.where(
            columns < n_vertices[:, None],
            (columns + 1) % n_vertices[:, None],
            columns,
        )
        next_next_col = np.where(
            columns < n_vertices[:, None],
            (columns + 2) % n_vertices[:, None],
            columns,
        )

        v_next = vertices[rows, next_col]
        v_next_next = vertices[rows, next_next_col]
        del rows, next_col, next_next_col

        edges1 = v_next - vertices
        edges2 = v_next_next - v_next
        del v_next, v_next_next

        # Work out whether each cell is convex
        if self.spherical:
            valid_3d = ~mask[:, :, None]
            centroids = (vertices * valid_3d).sum(axis=1) / n_vertices[:, None]
            del valid_3d, vertices
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

            crosses_3d = np.cross(edges1, edges2)
            del edges1, edges2

            dots = (crosses_3d * centroids[:, None, :]).sum(axis=2)
            del crosses_3d, centroids

            anticlockwise = np.all((dots > atol) | mask, axis=1)
            clockwise = np.all((dots < -atol) | mask, axis=1)
            del dots
        else:
            del vertices

            crosses_2d = (
                edges1[:, :, 0] * edges2[:, :, 1]
                - edges1[:, :, 1] * edges2[:, :, 0]
            )
            del edges1, edges2

            anticlockwise = np.all((crosses_2d > atol) | mask, axis=1)
            clockwise = np.all((crosses_2d < -atol) | mask, axis=1)
            del crosses_2d

        is_convex = anticlockwise | clockwise
        del anticlockwise, clockwise

        # Exclude degenerate cells
        is_convex &= n_vertices >= 3

        return is_convex

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
                "Can't locate points that need converting from latlon: "
                "Must install the 'pyproj' library"
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

    def _ear_clip(self, face_vertices, atol):
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
            return sign * cross_scalar(v1, v2) > atol

        def contains_no_other_points(a, b, c, idx_list):
            pa, pb, pc = face_vertices[a], face_vertices[b], face_vertices[c]
            for p_idx in idx_list:
                if p_idx in (a, b, c):
                    continue

                pt = face_vertices[p_idx]
                c1 = sign * cross_scalar(pb - pa, pt - pa)
                c2 = sign * cross_scalar(pc - pb, pt - pb)
                c3 = sign * cross_scalar(pa - pc, pt - pc)
                if c1 >= -atol and c2 >= -atol and c3 >= -atol:
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

    def _point_in_triangle(self, triangle, point, anticlockwise, atol):
        """True if a point is inside a triangle.

        .. versionadded:: NEXTVERSION

        :Parameters:

            triangle: array_like

                The coordinates of the triangle's vertices. Must have
                shape ``(3, N)`` for N-d coordinates.

            point: array_like

                The coordinates of the point to test for containment
                in the triangle. Must have shape ``(N,)`` for N-d
                coordinates.

            anticlockwise: `bool`

                True if the triangle vertices are ordered
                anticlockwise, otherwise False.

              atol: `float` TODO

        :Returns:

                True if the point is inside the triangle, otherwise
                False.

        """
        if self.spherical:
            crosses = np.cross(triangle, np.roll(triangle, -1, axis=0))
            values = np.dot(crosses, point)
            del crosses
        else:
            t_next = np.roll(triangle, -1, axis=0)
            crosses = (t_next[:, 0] - triangle[:, 0]) * (
                point[1] - triangle[:, 1]
            ) - (t_next[:, 1] - triangle[:, 1]) * (point[0] - triangle[:, 0])
            values = crosses

        if anticlockwise:
            is_inside = np.all(values >= -atol)
        else:
            is_inside = np.all(values <= atol)

        return is_inside

    def locate(self, y, x, latlon=True, unique=True, k_initial=12):
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

        TODO

        Grid cells are internally decomposed into non-overlapping
        sub-triangles. For each target point, the *k_initial*
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
                units of degrees north within the domain's (implied)
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
                units of degrees east within the domain's (implied)
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
                without duplicates, and without placeholders for
                points that are out-of-domain (i.e that do not lie
                inside any cell). If False then an index is returned
                for every point, including placeholders for
                out-of-domain points.

                The out-of-domain placeholder is ``-N``, where ``N``
                is the size of the discrete axis, which is always a
                non-valid index.

            k_initial: `int`, optional

                The number of nearest sub-triangle centroids to query
                in the initial k-d tree search when locating a point.
                Grid cells are internally decomposed into
                non-overlapping sub-triangles. For each point to
                locate, the *k_initial* sub-triangles whose centroids
                are closest to the point are retrieved and tested for
                containment. If no match is found from those
                sub-triangles, and the point has not been identified as
                being out-of-domain, then sub-triangles with more
                distant centroids are progressively included until a
                match is found or the point is confirmed not to lie in
                any cell.

                By default, *k_initial* is ``12``. This is usually
                sufficient for locating, without needing to widen the
                search radius, any point in a mesh composed of
                low-aspect-ratio triangles, quadrilaterals, pentagons,
                or hexagons (because for such meshes it is likely both
                that the point lies in one of the face cells which
                share the mesh node closest to the point, and that
                there are at most twelve sub-triangles which share
                that same mesh node), but smaller or larger numbers
                may give better performance in individual cases.

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
            return np.array([], dtype=np.int32)

        y = y.astype(np.float64)
        x = x.astype(np.float64)

        # Initialise the output array of indices. It is important that
        # it is initialised to -self.n_cells (i.e. a negative value
        # that is not a valid index of the cells).
        n_points = y.size
        result = np.full(
            n_points, -self.n_cells, dtype=integer_dtype(self.n_cells)
        )

        n_triangles = len(self.triangle_nodes)
        if not n_triangles:
            # There are only degenerate cells, so no points can lie in
            # them
            return result

        # Parse k_initial
        k_start = 0
        k_stop = int(k_initial)
        if k_stop < 1:
            raise ValueError(
                f"k_initial must be a positive integer. Got: {k_initial!r}"
            )

        # ------------------------------------------------------------
        # Parse the y and x points
        # ------------------------------------------------------------
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

        # Initialise the list of points that have not been resolved,
        # i.e. those points for which it is not yet known whether or
        # not they lie inside a cell.
        unresolved_points = np.arange(n_points, dtype=integer_dtype(n_points))

        # Get the theoretical maximum distance that a point can be
        # from a sub-triangle centroid and still be inside the parent
        # cell. I.e. if a point is at least this far away from a
        # sub-triangle centroid, then the point cannot lie in the
        # parent cell.
        max_cell_radius = self.max_cell_radius
        count = 0
        while len(unresolved_points) > 0:
            Q_unmatched = Q[unresolved_points]

            # --------------------------------------------------------
            # Query the kd-tree to find:
            #
            # distances: For each point, how far it is from its
            #            nearest triangle centroids[*] (ordered by
            #            proximity to the point). Encodes 'inf' for
            #            distances greater than or equal to the
            #            maximum cell radius.
            #
            # triangle_indices: For each point, the indices of the
            #                   triangles with the nearest
            #                   centroids[*] (ordered by proximity to
            #                   the point). Encodes 'n_triangles' for
            #                   distances greater than or equal to the
            #                   maximum cell radius.
            #
            # [*] "nearest" excludes even closer centroids that have
            #     already been checked and discounted in a previous
            #     iteration.
            #
            # --------------------------------------------------------
            k_stop = min(k_stop, n_triangles)
            distances, triangle_indices = self.kdtree.query(
                Q_unmatched,
                k=range(k_start + 1, k_stop + 1),
                distance_upper_bound=max_cell_radius,
            )

            # The number distances
            n_distances = len(distances)

            # List of points that lie inside a cell
            matched = []

            # List of points that do not lie inside any cell anywhere
            # in the domain
            out_of_domain = []

            for p_index, point_distances, triangles in zip(
                unresolved_points, distances, triangle_indices
            ):
                # The coordinates of the unresolved point (shape (N,))
                point = Q[p_index]

                if triangles[0] == n_triangles:
                    # The distance from the point to its closest
                    # triangle centroid is greater than or equal to
                    # the maximum cell radius => the point cannot be
                    # inside any cell anywhere in the domain, and we
                    # don't need to check anything else.
                    out_of_domain.append(p_index)
                    continue

                # Get rid of triangles for which the distances of
                # their centroids to the point are greater or equal to
                # the maximum cell radius (the point definitely can't
                # be in those).
                triangles = triangles[triangles < n_triangles]

                # Perform an expensive geometric point-in-triangle
                # test for each triangle, in order of proximity to the
                # point.
                point_is_in_triangle = False
                for t_index in triangles:
                    # Coordinates of this triangle's vertices (shape (3, N))
                    triangle = self.nodes[self.triangle_nodes[t_index]]
                    count += 1
                    if self._point_in_triangle(
                        triangle, point, True, self.atol
                    ):
                        # The point is inside this triangle => it lies
                        # inside a cell.
                        result[p_index] = self.parent_face_ids[t_index]
                        matched.append(p_index)
                        point_is_in_triangle = True
                        break

                if not point_is_in_triangle and len(triangles) < n_distances:
                    # The point is not in any triangle whose centroid
                    # is closer to the point than the maximum cell
                    # radius => the point cannot be inside any cell
                    # anywhere in the domain.
                    out_of_domain.append(p_index)

            resolved = matched + out_of_domain
            if resolved:
                # Remove resolved points from list of unresolved points
                unresolved_points = np.setdiff1d(
                    unresolved_points, resolved, assume_unique=True
                )

            if k_stop >= n_triangles:
                # We've checked everything!
                break

            # Still here? Then double the number of candidate
            # triangles for the next iteration.
            k_start = k_stop
            k_stop = k_stop + k_stop

        size = result.size
        if unique and size:
            # --------------------------------------------------------
            # Return sorted zero-based cell indices without
            # duplicates, nor placeholders for points that are
            # out-of-domain (i.e that do not lie inside any cell).
            # --------------------------------------------------------
            if size > 1:
                # Unique and sort
                result = np.unique(result)

            if result[0] < 0:
                # Remove the negative-valued out-of-domain placeholder
                result = result[1:]

        return result
