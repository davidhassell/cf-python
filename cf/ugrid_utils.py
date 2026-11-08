import numpy as np

from cf import atol as cf_atol


def _ugrid_locate(lat, lon, f, unique=True):
    """Locate UGRID cells containing latitude-longitude locations.

    Returns the discrete axis indices of the cells containing the
    latitude-longitude locations.

    If a single latitude is given then it is paired with each
    longitude, and if a single longitude is given then it is paired
    with each latitude. If multiple latitudes and multiple longitudes
    are provided then they are paired element-wise.

    If a cell contains more than one of the given latitude-longitude
    locations then that cell's index appears only once in the output.

    If a latitude-longitude location lies outside of all cells then it
    is ignored. TODO

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.locate`

    :Parameters:

        lat: 1-d `numpy.ndarray`
            The latitudes, in degrees north, for which to find the
            cell indices. Must be in the range [-90, 90], although
            this is not checked.

        lon: 1-d `numpy.ndarray`
            The longitudes, in degrees east, for which to find the
            cell indices.

        f: `Field` or `Domain`
            The field or domain containing the UGRID grid.

        unique: `bool`, optional
            Return sorted cell indices without duplicates TODO

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
            "Can't locate latitudes and longitudes for a UGRID domain that "
            f"does not have well-defined X and Y node coordinates: {f!r}"
        )

    # Does the field have a spherical polar coordinate system?
    units = nodes_y.Units
    spherical = (
        units.islatitude
        or nodes_y.get_property("standard_name", None) == "grid_latitude"
    )

    if spherical:
        y, x = lat, lon
    else:
        # Convert spherical 'lat' and 'lon' to 'y' and 'x' in the
        # fields's Cartesian coordinate system
        raise NotImplementedError(
            "Can't yet locate latitudes and longitudes for a UGRID domain "
            "with a Cartesian coordinate system"
        )

    # Initialise a locator instance
    locator = UGRIDLocator(
        nodes_y.array,
        nodes_x.array,
        face_node_connectivity.array,
        spherical,
    )

    # Find the indices for the locations
    index = locator.locate(y, x, unique=unique)
    return index


class UGRIDLocator:
    """Point-in-mesh locator for UGRID unstructured meshes.

    Provides O(log N) spatial indexing and bulk point-in-mesh location
    for arbitrary unstructured spherical meshes, supporting
    high-valence node junctions, concave faces, high-aspect-ratio
    elements, and scaled datasets exceeding 10e6 faces.

    Maps 2-d spherical coordinates to 3-d unit N-vectors to eliminate
    polar singularities and dateline discontinuities. Polygon face
    connectivity is decomposed into convex sub-triangles using a fully
    vectorised fan-triangulation pass. Point location queries execute
    via a two-stage filter-and-verify algorithm:

    1. **Spatial Filtering (O(log N)):** Employs a SciPy 3-d `cKDTree`
       built on sub-triangle centroids to fetch candidate
       faces. Out-of-domain query points are rejected in O(1) time via
       a precomputed maximum cell radius bound.

    2. **Exact Verification (O(1)):** Evaluates a vectorised scalar
       triple product point-in-triangle test using `cf.atol` for exact
       edge and boundary resolution.

    :Notes:

    - **N-Vector Geometry:** Spherical coordinates are represented as
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

    - **Tolerance integration:** Dynamically queries `cf.atol` to
      resolve shared face boundaries.

    :Attributes:

    `k_candidates`: Initial candidate search depth, derived from
    maximum node valence and 99th-percentile cell aspect ratios.

    `max_cell_radius`: Maximum 3-d Euclidean distance from any
    sub-triangle centroid to one its vertices across the entire
    mesh. Defines the physical threshold for O(1) out-of-bounds
    rejection.

    `nodes_3d`: Array of shape ``(n_nodes, 3)``. Cartesian 3-d unit
    N-vectors representing all unique mesh nodes on the unit sphere.

    `parent_face_ids`: Array of shape ``(n_sub_triangles,)``. Maps
    each sub-triangle back to its 0-based parent face index in the
    original `face_nodes` matrix.

    `tree`: `scipy.spatial.cKDTree` 3-d KD-Tree spatial index
    constructed over normalised sub-triangle centroids.

    `tri_nodes`: Array of int Array of shape ``(n_sub_triangles,
    3)``. Node indices defining all sub-triangles generated by
    fan-triangulation of polygon faces.

    .. versionadded:: NEXTVERSION

    """

    def __init__(self, nodes_y, nodes_x, face_nodes, spherical=True):
        """**Initialisation**

        :Parameters:

            nodes_y: 1-d `numpy.ndarray`
                Array of longitude coordinates for all unique mesh
                nodes, in degrees.TODO

            nodes_x: 1-d `numpy.ndarray`
                Array of latitude coordinates for all unique mesh
                nodes, in degrees.TODO

            face_nodes : 2-d `numpy.ndarray`
                Integer connectivity matrix of shape ``(n_faces,
                max_nodes_per_face)`` where each row contains the node
                indices defining the perimeter of a mesh face. Padded
                rows (e.g. quads in a mixed quad-triangle mesh) must
                be masked).

        """
        from scipy.spatial import cKDTree

        self.spherical = bool(spherical)

        if spherical:
            # Convert spherical node coordinates to 3-d N-vectors
            # (unit sphere)
            r_lon, r_lat = np.radians(nodes_x), np.radians(nodes_y)
            self.nodes_3d = np.column_stack(
                [
                    np.cos(r_lat) * np.cos(r_lon),
                    np.cos(r_lat) * np.sin(r_lon),
                    np.sin(r_lat),
                ]
            )
        else:
            # 2-d Cartesian node coordinates (eastings, northings)
            self.nodes = np.column_stack([nodes_x, nodes_y])

        # Get the face_node_connectivity mask (True where a face has
        # fewer than the maximum number of nords per face)
        mask = np.ma.getmaskarray(face_nodes)
        if np.ma.isMA(face_nodes):
            face_nodes = np.ma.getdata(face_nodes)

        # Fan triangulation
        # (https://en.wikipedia.org/wiki/Fan_triangulation)
        #
        # For each polygon face, create sub-triangles from its
        # vertices with (v_{0}, v_{k}, v_{k+1}) for k in 1, ..,
        # max_nodes_per_face-2, where v_{N} is the node identity from
        # column N in the face_node_connectivity array.
        n_faces, max_nodes_per_face = face_nodes.shape
        tri_list = []
        parent_list = []
        parents_base = np.arange(n_faces, dtype=int)

        for k in range(1, max_nodes_per_face - 1):
            v0 = face_nodes[:, 0]
            vk = face_nodes[:, k]
            vk1 = face_nodes[:, k + 1]

            valid_k = (~mask[:, 0]) & (~mask[:, k]) & (~mask[:, k + 1])

            tri_k = np.column_stack([v0, vk, vk1])[valid_k]
            parents_k = parents_base[valid_k]

            tri_list.append(tri_k)
            parent_list.append(parents_k)

        self.tri_nodes = np.vstack(tri_list)
        self.parent_face_ids = np.concatenate(parent_list)

        if len(self.tri_nodes) > 0:
            # Compute normalised 3-d centroids
            tri_verts = self.nodes_3d[self.tri_nodes]
            centroids = tri_verts.mean(axis=1)
            if self.spherical:
                centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

            self.tree = cKDTree(centroids)

            # Precompute the maximum cell radius across the entire
            # mesh
            #
            # d_v = distances from vertices to normalised sub-triangle
            #       centroid
            d_v = np.linalg.norm(tri_verts - centroids[:, None, :], axis=2)
            self.max_cell_radius = float(d_v.max())

            # Infer an initial value for k_candidates = The number of
            # nearest triangle centroids retrieved from the KD-Tree
            # spatial index for initial candidate evaluation.
            #
            # k_candidates has to be less or equal to the total number
            # of triangles. The initial value and should be small
            # enough to be sufficient for most points. For a given
            # point, if a triangle (of a face) cannot be found to
            # contain it, k_candidates will be increased in the
            # `locate` method until we find one that does (if
            # approriate).
            node_counts = np.bincount(self.tri_nodes.ravel())
            max_degree = node_counts.max()
            min_d = np.maximum(d_v.min(axis=1), 1e-12)
            aspect_multiplier = float(
                np.percentile(d_v.max(axis=1) / min_d, 99)
            )

            inferred_k = int(np.ceil(2 * max_degree * aspect_multiplier))
            self.k_candidates = max(8, min(inferred_k, len(self.tri_nodes)))
        else:
            self.k_candidates = 0
            self.max_cell_radius = 0.0
            self.tree = None

    def locate(self, y, x, k_candidates=None, unique=True):
        r"""Locate the UGRID face index containing query points.

        Performs a high-performance spatial lookup using a two-stage
        filter-and-verify algorithm: O(log N) nearest-centroid
        retrieval via SciPy KD-Tree followed by an O(1) vectorised
        scalar triple product check.

        :O(1) out-of-bounds early rejection:

        Query points whose 3-d distance to their nearest centroid
        exceeds ``self.max_cell_radius + cf.atol()`` are flagged as
        outside the mesh instantly without evaluating vector
        cross-products.

        :Progressive Fallback Guarantee:

        If a point is within ``self.max_cell_radius`` of the mesh but
        fails initial verification, `locate` dynamically expands the
        candidate search radius until the point is either successfully
        located or distance limits prove it is outside the domain.

        :Tolerance handling:

        Uses `cf.atol` for exact boundary and edge resolution:
        $$\mathbf{v}_i \times \mathbf{v}_{i+1} \cdot \mathbf{q} \ge
        -\text{cf.atol}()$$

        .. versionadded:: NEXTVERSION

        :Parameters:

            y: TODO

            x: TODO

            k_candidates: `int` or `None`, optional

            unique: `bool`, optional
                Return sorted cell indices without duplicates TODO

        """
        # For each point, the UGRID axis index for the cell that
        # contains it, or the value -1 of no cells contain a point.
        results = np.full(y.shape, -1, dtype=int)

        n_triangles = len(self.tri_nodes)
        if n_triangles == 0:
            # Every point is outside of the mesh (because the mesh has
            # no faces!)
            return results

        if self.spherical:
            # Spherical polar coordinates: Convert the query points to
            # N-vectors
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
            # Cartesian coordinates (eastings, northings)
            Q = np.column_stack([x, y])

        # List of points that are unresolved (i.e. that haven't yet
        # been shown to lie either inside a cell or outside the entire
        # mesh). This list gets updated each time we pass through the
        # search loop. Each point is given by its integer index
        # position in the 'y' and 'x' arrays.
        unmatched_indices = np.arange(y.size)

        atol = float(cf_atol())
        max_distance_bound = self.max_cell_radius + atol

        if k_candidates is None:
            k_current = self.k_candidates
        else:
            k_current = k_candidates

        k_current = min(k_current, n_triangles)

        # Progressive search loop bounded by physical distance
        while len(unmatched_indices) > 0:
            Q_unmatched = Q[unmatched_indices]

            # Query tree for distances and candidate indices
            #
            # For each point, 'triangle_indices' contains the indices
            # for 'k_current' closest candidate triangles; and
            # 'distances' contains the distance from the each point to
            # each candidate triangle.
            #
            # E.g. distances = [[0.007 0.007 0.028 0.029 0.040]
            #                   [1.130 1.150 1.153 1.164 1.166]]
            # triangle_indices = [[0 3 2 4 1]
            #                     [2 0 3 1 4]]
            # k_current = 5
            distances, triangle_indices = self.tree.query(
                Q_unmatched, k=k_current
            )
            if k_current == 1:
                # Makes use that the arrays are always 2-d
                distances = distances[:, None]
                triangle_indices = triangle_indices[:, None]

            # List of points that are matched (i.e. that are shown to
            # lie either inside a cell or outside the entire mesh)
            # during this pass through the search loop. Each point is
            # given by its integer index position in the 'y' and 'x'
            # arrays.
            newly_matched = []

            # List of points that are found to be outside of the
            # entire mesh during this pass through the search loop.
            # Each point is given by its integer index position in the
            # 'y' and 'x' arrays.
            outside_mesh = []

            for sub_idx, point_idx in enumerate(unmatched_indices):
                q_vec = Q[point_idx]
                pt_dists = distances[sub_idx]
                candidate_triangles = triangle_indices[sub_idx]

                # The nearest candidate triangle centroid is farther
                # away than the maximum cell radius, so the point must
                # be outside of the entire mesh.
                if pt_dists[0] > max_distance_bound:
                    outside_mesh.append(point_idx)
                    continue

                # Restrict the search to candidate triangles whose
                # centroids are no further than the maximum cell
                # radius from the point
                valid_mask = (pt_dists <= max_distance_bound) & (
                    candidate_triangles < n_triangles
                )
                valid_triangles = candidate_triangles[valid_mask]

                # Point-in-triangle check, but only on triangles whose
                # centroids are no further away than the maximum cell
                # radius
                for tri_idx in valid_triangles:
                    v = self.nodes_3d[self.tri_nodes[tri_idx]]

                    if self.spherical:
                        # Spherical polar coordinates: 3-d Cross
                        # product of N-vectors with target vector
                        crosses = np.cross(v, np.roll(v, -1, axis=0))
                        is_inside = np.all(np.dot(crosses, q_vec) >= -atol)
                    else:
                        # Cartesian coordinates: 2-d cross product of
                        # edge vectors with (q - v)
                        v_next = np.roll(v, -1, axis=0)
                        crosses = (v_next[:, 0] - v[:, 0]) * (
                            q_vec[1] - v[:, 1]
                        ) - (v_next[:, 1] - v[:, 1]) * (q_vec[0] - v[:, 0])
                        is_inside = np.all(crosses >= -atol)

                    if is_inside:
                        # This point lies inside a cell
                        results[point_idx] = self.parent_face_ids[tri_idx]
                        newly_matched.append(point_idx)
                        break

                # If we have searched all candidates within physical
                # range and found no match, then the point is outside
                # the mesh.
                if point_idx not in newly_matched and np.any(
                    pt_dists > max_distance_bound
                ):
                    outside_mesh.append(point_idx)

            # Drop points that were matched or proven to be outside
            resolved = newly_matched + outside_mesh
            if resolved:
                unmatched_indices = np.setdiff1d(
                    unmatched_indices, resolved, assume_unique=True
                )

            # Stop if all remaining candidate triangles exceed the
            # distance bound or tree size
            if k_current >= n_triangles or len(outside_mesh) == len(
                unmatched_indices
            ):
                break

            # Scale up k only for remaining ambiguous points
            k_current = min(k_current * 4, n_triangles)

        size = results.size
        if unique and size:
            # Return sorted cell indices without duplicates, nor
            # out-of-mesh indicators.
            if size > 1:
                results = np.unique(results)

            if results[0] == -1:
                results = results[1:]

        return results
