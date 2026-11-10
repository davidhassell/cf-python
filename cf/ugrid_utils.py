import numpy as np

from cf import atol as cf_atol


def _ugrid_locate(y,x, f, latlon=True,unique=True, return_operator=False):
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

        lat: array-like or `None`
            The latitudes, in degrees north, for which to find the
            cell indices. `None` is equalalent to an empty list.

        lon: array-like or `None`
            The longitudes, in degrees east, for which to find the
            cell indices. `None` is equalalent to an empty list.

        f: `Field` or `Domain`
            The field or domain containing the UGRID grid.

        unique: `bool`, optional
            If True (the default) then return sorted cell indices
            without duplicates, nor placeholders for locations that do
            not lie inside any cell. If False then an index is
            returned for every location, including non-valid index
            placeholders for locations that do not lie inside any
            cell.

            The outside-of-mesh placeholder is ``-N``, where ``N`` is
            the size of the UGRID discrete axis, which is always a
            non-valid index.

        return_operator: `bool`, optional
            If True then do not locate cells, but return an operator
            that is able to do the locating. The operator is a
            function that takes the same *lat*, *lon* and *unique*
            arguments as this function, and also has extra
            configuration parameters. When multiple `cf.locate` calls
            for the same field or domain are expected, using the
            operator to do the locating can be much faster.

            When returning the operator, the *lat* and *lon*
            parameters are ignored (so may both be set to `None`), and
            the *unique* parameter is also ignored.

    :Returns:

        `numpy.ndarray`
            Indices for the discrete UGRID axis that contain the
            latitude-longitude locations.

    """
    # Get the face_node_connectivity array, and normalise it.
    face_node_connectivity = f.domain_topology(filter_by_cell=("face",))
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

    # Does the field have a spherical polar node coordinates?
    spherical = nodes_y.Units.islatitude and nodes_x.Units.islongitude

    # Still here?
    if not spherical:
        raise NotImplementedError(
            "Can't yet locate points for a UGRID domain with a "
            "non-latitude_longitude coordinate system"
        )

    # Initialise a polygon locator instance
    locator = PolygonLocator(
        nodes_y.array,
        nodes_x.array,
        face_node_connectivity.array,
        spherical=spherical,
        atol=cf_atol(),
    )

    if return_operator:
        return locator.locate

    # Still here?
    if not spherical:
        # Convert spherical polar 'lat' and 'lon' to 'y' and 'x' in
        # the domain's non-latitude_longitude (e.g. Cartesian,
        # rotated_latitude_longitude) coordinate system.
        raise NotImplementedError(
            "Can't yet locate latitudes and longitudes for a UGRID "
            "domain with a non-latitude_longitude coordinate system"
        )

    # Find the indices for the points
    index = locator.locate(y, x, latlon=latlon,unique=unique)
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

    `tri_nodes`: Array of int Array of shape ``(n_sub_triangles,
    3)``. Node indices defining all sub-triangles generated by
    fan-triangulation of polygon faces.

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.locate`

    """

    def __init__(
            self, nodes_y, nodes_x, face_nodes, cr, latlon_cr, atol=1e-12,
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

            grid_mapping:

            latlon_grid_mapping:
        
            spherical: `bool`, optional
                If True (the default) then coordinates are spherical
                lat/lon. If False, coordinates are 2-d Cartesian plane
                coordinates.

            atol: `float`, optional
                Absolute tolerance for floating point comparisons.

        """
        from scipy.spatial import cKDTree
        
        self.cr = cr
        self.latlon_cr = latlon_cr
        self.atol = float(atol)

        self.spherical = self.grid_mapping_name(cr) in (
            "latitude_longitude", "rotated_latitude_longitude" )

        # Build node coordinates array
        if spherical:
            # Spherical polar node coordinates (possibly rotated):
            # Convert to 3-d N-vectors. Array shape is (n_nodes, 3).
            nodes_x, nodes_y = np.radians(nodes_x), np.radians(nodes_y)
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

        tri_list = []
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

            tri_list.append(tri_k)
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
                tri_list.append(np.array(local_tri_nodes, dtype=int))
                parent_list.append(
                    np.full(len(local_tris), face_idx, dtype=int)
                )

        del mask, local_tri_nodes, local_tris

        # Build sub-triangle data structures and cKDTree spatial index
        if tri_list:
            tri_nodes = np.vstack(tri_list)
        else:
            tri_nodes = np.empty((0, 3), dtype=int)

        del tri_list

        if len(tri_nodes) > 0:
            self.tri_nodes = np.vstack(tri_nodes)
            del tri_nodes

            self.parent_face_ids = np.concatenate(parent_list)
            del parent_list

            tri_verts = self.nodes[self.tri_nodes]
            centroids = tri_verts.mean(axis=1)
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
            d_v = np.linalg.norm(tri_verts - centroids[:, None, :], axis=2)
            del centroids, tri_verts

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
            node_counts = np.bincount(self.tri_nodes.reshape(-1))
            max_degree = node_counts.max()
            del node_counts

            min_d = np.maximum(d_v.min(axis=1), self.atol)
            aspect_multiplier = float(
                np.percentile(d_v.max(axis=1) / min_d, 99)
            )
            del d_v, min_d

            inferred_k = int(np.ceil(2 * max_degree * aspect_multiplier))
            self.k_candidates = max(8, min(inferred_k, len(self.tri_nodes)))
        else:
            self.tri_nodes = np.empty((0, 3), dtype=int)
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

                    if point_in_triangle(polygon[other_i], p_prev, p_curr, p_next):
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
        r"""Find the indices UGRID face cells containing locations.

        Performs a high-performance spatial lookup using a two-stage
        filter-and-verify algorithm: O(log N) nearest-centroid
        retrieval via KD-Tree followed by an O(1) vectorised scalar
        triple product check.

        :O(1) out-of-bounds early rejection:

        Query points whose 3-d distance to their nearest centroid
        exceeds ``self.max_cell_radius + self.atol`` are flagged as
        outside the mesh instantly without evaluating vector
        cross-products.

        :Progressive fallback guarantee:

        If a point is within ``self.max_cell_radius`` of the mesh but
        fails initial verification, `locate` dynamically expands the
        candidate search radius until the point is either successfully
        located or distance limits prove it is outside the domain.

        :Tolerance handling:

        Uses `self.atol` for exact boundary and edge resolution:
        $$\mathbf{v}_i \times \mathbf{v}_{i+1} \cdot \mathbf{q} \ge
        -\text{self.atol}$$

        .. versionadded:: NEXTVERSION

        .. seealso:: `cf.locate`

        :Parameters:

            y: array-like or `None`
                The latitudes, in degrees north, for which to find the
                cell indices. `None` is equalalent to an empty list.

            x: array-like or `None`
                The longitudes, in degrees east, for which to find the
                cell indices. `None` is equalalent to an empty list.

            unique: `bool`, optional
                If True (the default) then return sorted cell indices
                without duplicates, nor non-valid index placeholders
                for locations that do not lie inside any cell. If
                False then an index is returned for every location,
                including non-valid index placeholders for locations
                that do not lie inside any cell.

                The non-valid index placeholder is ``-N``, where N is
                the size of the UGRID discrete axis.

            k_candidates: `int` or `None`, optional

            _cartesian: `bool`, optional
                *y* and *x* are instead Cartesian coordinates with the
                same units as the Y and X node coodinates of the
                parent domain. This is feature is only for testing,
                and doesn't come with any checks on the consistency
                between *y* and *x* and the node coordinates.

        :Returns:

            1-d `numpy.ndarray`
                Indices for the HEALPix axis that contain the
                latitude-longitude locations. See the *unique*
                parameter.

        """
        if self.grid_mapping == "latitude_longitude":pass
        
        
        if self.spherical and _cartesian:
            raise ValueError(
                "Can't set _cartesian=True when self.spherical=True"
            )

        if not self.spherical and not _cartesian:
            raise NotImplementedError(
                "Can't yet locate latitudes and longitudes for a UGRID "
                "domain with a non-latitude_longitude coordinate system"
            )

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

        n_triangles = len(self.tri_nodes)
        if n_triangles == 0:
            return results

        if self.grid_mapping_name(self.cr) == "latitude_longitude":
            if not latlon:
                # Nodes are defined in latitude_longitude, but the
                # points are not.
                raise ValueError("TODO")
        elif latlon:
            # Points are defined in latitude_longitude, but the nodes
            # are not => Convert the points to the CRS of the nodes.
            #
            # Make use of self.cr and self.latlon_cr
            pass
        
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
            r_lat = np.radians(y)
            r_lon = np.radians(x)
            Q = np.column_stack(
                [
                    np.cos(r_lat) * np.cos(r_lon),
                    np.cos(r_lat) * np.sin(r_lon),
                    np.sin(r_lat),
                ]
            )
        else:
            # 2-d Cartesian point coordinates. Array shape is
            # (n_nodes, 2).
            Q = np.column_stack([x, y])

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
                    v = self.nodes[self.tri_nodes[tri_idx]]

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
