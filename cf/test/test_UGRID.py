import atexit
import datetime
import faulthandler
import itertools
import os
import tempfile
import unittest

import netCDF4
import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf

warnings = False

# Set up temporary files
n_tmpfiles = 2
tmpfiles = [
    tempfile.mkstemp("_test_ugrid.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
[tmpfile, tmpfile1] = tmpfiles


def _remove_tmpfiles():
    """Remove temporary files created during tests."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


def n_mesh_variables(filename):
    """Return the number of mesh variables in the file."""
    nc = netCDF4.Dataset(filename, "r")
    n = 0
    for v in nc.variables.values():
        try:
            v.getncattr("topology_dimension")
        except AttributeError:
            pass
        else:
            n += 1

    nc.close()
    return n


def combinations(face, edge, point):
    """Return combinations for field/domain indexing."""
    return [
        i
        for n in range(1, 4)
        for i in itertools.permutations([face, edge, point], n)
    ]


class UGRIDTest(unittest.TestCase):
    """Test UGRID field constructs."""

    filename1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_1.nc"
    )

    filename2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_2.nc"
    )

    filename3 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_3.nc"
    )

    global_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_global_1.nc"
    )

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows: cf.LOG_LEVEL('DEBUG')
        #
        # < ... test code ... >
        # cf.log_level('DISABLE')

    def test_UGRID_read(self):
        """Test reading of UGRID files."""
        f1 = cf.read(self.filename1)

        self.assertEqual(len(f1), 3)
        for g in f1:
            self.assertEqual(len(g.domain_topologies()), 1)
            self.assertEqual(len(g.auxiliary_coordinates()), 2)
            self.assertEqual(len(g.dimension_coordinates()), 1)

            for aux in g.auxiliary_coordinates().values():
                self.assertTrue(aux.has_data())

            if g.domain_topology().get_cell() == "face":
                self.assertEqual(len(g.cell_connectivities()), 1)
                self.assertEqual(
                    g.cell_connectivity().get_connectivity(), "edge"
                )

        f2 = cf.read(self.filename2)
        self.assertEqual(len(f2), 3)
        for g in f2:
            self.assertEqual(len(g.domain_topologies()), 1)
            self.assertEqual(len(g.auxiliary_coordinates()), 2)
            self.assertEqual(len(g.dimension_coordinates()), 1)

            cell = g.domain_topology().get_cell()
            if cell in ("edge", "face"):
                for aux in g.auxiliary_coordinates().values():
                    self.assertFalse(aux.has_data())

            if cell == "face":
                self.assertEqual(len(g.cell_connectivities()), 1)
                self.assertEqual(
                    g.cell_connectivity().get_connectivity(), "edge"
                )

    def test_UGRID_data(self):
        """Test reading of UGRID data."""
        node1, face1, edge1 = cf.read(self.filename1)
        node2, face2, edge2 = cf.read(self.filename2)

        # Domain topology arrays
        domain_topology1 = face1.domain_topology()
        self.assertTrue(
            (
                domain_topology1.array
                == np.array([[2, 3, 1, 0], [4, 5, 3, 2], [1, 3, 6, -99]])
            ).all()
        )
        self.assertTrue(domain_topology1.equals(face2.domain_topology()))

        domain_topology1 = edge1.domain_topology()
        self.assertTrue(
            (
                domain_topology1.array
                == np.array(
                    [
                        [1, 6],
                        [3, 6],
                        [3, 1],
                        [0, 1],
                        [2, 0],
                        [2, 3],
                        [2, 4],
                        [5, 4],
                        [3, 5],
                    ]
                )
            ).all()
        )
        self.assertTrue(domain_topology1.equals(edge2.domain_topology()))

        # Cell connectivity arrays
        cell_connectivity1 = face1.cell_connectivity()
        self.assertTrue(
            (
                cell_connectivity1.array
                == np.array(
                    [
                        [0, 1, 2, -99, -99],
                        [1, 0, -99, -99, -99],
                        [2, 0, -99, -99, -99],
                    ]
                )
            ).all()
        )
        self.assertTrue(cell_connectivity1.equals(face2.cell_connectivity()))

    def test_read_UGRID_domain(self):
        """Test reading of UGRID files into domains."""
        d1 = cf.read(self.filename1, domain=True)

        self.assertEqual(len(d1), 3)
        for g in d1:
            self.assertIsInstance(g, cf.Domain)
            self.assertEqual(len(g.domain_topologies()), 1)
            self.assertEqual(len(g.auxiliary_coordinates()), 2)
            self.assertEqual(len(g.dimension_coordinates()), 0)

            for aux in g.auxiliary_coordinates().values():
                self.assertTrue(aux.has_data())

            if g.domain_topology().get_cell() == "face":
                self.assertEqual(len(g.cell_connectivities()), 1)
                self.assertEqual(
                    g.cell_connectivity().get_connectivity(), "edge"
                )

    def test_read_write_UGRID_field(self):
        """Test the cf.read and cf.write with UGRID fields."""
        # Face, edge, and point fields that are all part of the same
        # UGRID mesh
        ugrid = cf.example_fields(8, 9, 10)
        face, edge, point = (0, 1, 2)

        tmpfile = "tmpfileu.nc"
        # Test for equality with the fields defined in memory. Only
        # works for face and edge fields.
        for cell in (face, edge):
            f = ugrid[cell]
            cf.write(f, tmpfile)
            g = cf.read(tmpfile)
            self.assertEqual(len(g), 1)
            self.assertTrue(g[0].equals(f))

        # Test round-tripping of field combinations
        for cells in combinations(face, edge, point):
            f = []
            for cell in cells:
                f.append(ugrid[cell])

            cf.write(f, tmpfile)

            # Check that there's only one mesh variable in the file
            self.assertEqual(n_mesh_variables(tmpfile), 1)

            g = cf.read(tmpfile)
            self.assertEqual(len(g), len(f))

            cf.write(g, tmpfile1)

            # Check that there's only one mesh variable in the file
            self.assertEqual(n_mesh_variables(tmpfile1), 1)

            h = cf.read(tmpfile1)
            self.assertEqual(len(h), len(g))
            self.assertTrue(h[0].equals(g[0]))

    def test_read_write_UGRID_domain(self):
        """Test the cf.read and cf.write with UGRID domains."""
        # Face, edge, and point fields/domains that are all part of
        # the same UGRID mesh
        ugrid = [f.domain for f in cf.example_fields(8, 9, 10)]
        face, edge, point = (0, 1, 2)

        # Test for equality with the fields defined in memory. Only
        # works for face and edge domains.
        for cell in (face, edge):
            d = ugrid[cell]
            cf.write(d, tmpfile)
            e = cf.read(tmpfile, domain=True)
            self.assertEqual(len(e), 2)
            self.assertTrue(e[0].equals(d))
            self.assertEqual(e[1].domain_topology().get_cell(), "point")

        # Test round-tripping of domain combinations for the
        # example_field domains, and also the domain read from
        # 'ugrid_3.nc'.
        for iteration in ("memory", "file"):
            for cells in combinations(face, edge, point):
                d = []
                for cell in cells:
                    d.append(ugrid[cell])

                if point not in cells:
                    # When we write a non-point domains, we also get
                    # the point locations.
                    d.append(ugrid[point])
                elif cells == (point,):
                    # When we write a point domain on its own, we also
                    # get the edge location.
                    d.append(ugrid[edge])

                cf.write(d, tmpfile)

                # Check that there's only one mesh variable in the file
                self.assertEqual(n_mesh_variables(tmpfile), 1)

                e = cf.read(tmpfile, domain=True)

                self.assertEqual(len(e), len(d))

                cf.write(e, tmpfile1)

                # Check that there's only one mesh variable in the file
                self.assertEqual(n_mesh_variables(tmpfile1), 1)

                f = cf.read(tmpfile1, domain=True)
                self.assertEqual(len(f), len(e))
                for i, j in zip(f, e):
                    self.assertTrue(i.equals(j))

            # Set up for the 'file' iteration
            ugrid = cf.read(self.filename3, domain=True)
            face, edge, point = (2, 1, 0)

            # Get the new face, edge, point order indices for the
            # domains read from disk (don't hard-wire these, as
            # implementation choices in netcdfread.py might change the
            # order :))
            for i, u in enumerate(ugrid):
                if u.domain_topology().get_cell() == "point":
                    point = i
                elif u.domain_topology().get_cell() == "edge":
                    edge = i
                elif u.domain_topology().get_cell() == "face":
                    face = i

    def test_UGRID_locate_1(self):
        """Test cf.locate with a limited area mesh."""
        f = cf.example_field(8)

        # Get the location for every point in the mesh
        node_lat = f.construct("latitude").array[::-1]
        node_lon = f.construct("longitude").array[::-1]
        self.assertEqual(cf.locate(node_lat, node_lon, f).tolist(), [0, 1, 2])

        # Get the indices for a subset of points in the mesh
        lat = [34, 32]
        lon = [-44, -44]
        self.assertEqual(cf.locate(lat, lon, f).tolist(), [0, 1])

        # Get the index for a point outside of the mesh
        lat = -89
        lon = 180
        self.assertEqual(cf.locate(lat, lon, f).tolist(), [])
        self.assertEqual(cf.locate(lat, lon, f, unique=False).tolist(), [-3])

        # Get the indices for a subset of points in the mesh and a
        # point outside of the mesh, including duplicates
        lat = [34, 32, -89, 34.1]
        lon = [-44, -44, 180, -44.1]
        self.assertEqual(cf.locate(lat, lon, f).tolist(), [0, 1])
        self.assertEqual(
            cf.locate(lat, lon, f, unique=False).tolist(),
            [0, 1, -3, 0],
        )

    def test_UGRID_locate_2(self):
        """Test cf.locate with a real C12 mesh."""
        # Locate global file path within cf-python test directory
        f = cf.read(self.global_file)[0]

        # Get the location for every point in the mesh
        node_lat = f.construct("latitude").array[::-1]
        node_lon = f.construct("longitude").array[::-1]

        self.assertEqual(
            cf.locate(node_lat, node_lon, f).tolist(),
            list(range(node_lat.size)),
        )
        self.assertEqual(
            cf.locate(node_lat, node_lon, f, unique=False).tolist(),
            list(range(node_lat.size - 1, -1, -1)),
        )

        # Now subspace the mesh
        N = 100
        index = slice(0, N)
        g = f.subspace(X=index)
        self.assertEqual(
            cf.locate(node_lat, node_lon, g).tolist(),
            cf.locate(node_lat, node_lon, f).tolist()[index],
        )

        result = cf.locate(node_lat, node_lon, g, unique=False).tolist()
        self.assertEqual(
            result[: node_lat.size - N], [-N] * (node_lat.size - N)
        )
        self.assertEqual(
            result[node_lat.size - N :], list(range(N - 1, -1, -1))
        )

    def test_UGRID_locate_3(self):
        """Test cf.locate with a concave cells."""
        f = cf.example_field(8)

        xbounds = f.coordinate("X").bounds
        xbounds[0, 1] = -44.5
        xbounds[1, 2] = -44.5
        xbounds[2, 2] = -44.5

        ybounds = f.coordinate("Y").bounds
        ybounds[0, 1] = 34.5
        ybounds[1, 2] = 34.5
        ybounds[2, 2] = 34.5

        x = f.coordinate("X")
        x[...] = [-44.9, -44, -43]

        y = f.coordinate("Y")
        y[...] = [34.9, 31.1, 34.5]

        # Get the location for every point in the mesh
        node_lat = f.construct("latitude").array[::-1]
        node_lon = f.construct("longitude").array[::-1]
        self.assertEqual(cf.locate(node_lat, node_lon, f).tolist(), [0, 1, 2])

        # Get the indices for a subset of points in the mesh and a
        # point outside of the mesh, including duplicates
        lat = [34.9, 34.4]
        lon = [-44.9, -44.4]
        self.assertEqual(
            cf.locate(lat, lon, f, unique=False).tolist(), [0, -3]
        )

        # An upside down "U" shaped cell interlocking with a "W" a
        # shaped cell. The numbers in bracket at the vertices are the
        # node indices.
        #
        #
        # Y=400           (12)----------------------(13)
        #                  |                          |
        #                  |       Face 1 ("U")       |
        #                  |                          |
        # Y=300  (0)------(11)     (8)------(7)      (4)------(3)
        #         |        |        |        |        |        |
        #         |        |        |        |        |        |
        #         |        |        |        |        |        |
        # Y=200   |       (10)-----(9)      (6)------(5)       |
        #         |                                            |
        #         |                Face 0 ("W")                |
        #         |                                            |
        # Y=100  (1)------------------------------------------(2)
        #
        #      X=100    X=200    X=300    X=400    X=500    X=600
        #
        # Test some left-to-right rows of points that cut through the
        # mesh at various Y values.
        from cf.ugrid_utils import PolygonWindow

        locator = PolygonWindow(
            nodes_y=[
                300,
                100,
                100,
                300,
                300,
                200,
                200,
                300,
                300,
                200,
                200,
                300,
                400,
                400,
            ],
            nodes_x=[
                100,
                100,
                600,
                600,
                500,
                500,
                400,
                400,
                300,
                300,
                200,
                200,
                200,
                500,
            ],
            face_nodes=np.ma.masked_equal(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # W
                    [12, 10, 9, 8, 7, 6, 5, 13, -1, -1, -1, -1],  # U
                ],
                -1,
            ),
            cr="Cartesian",  # Dummy, non-latitude_longitude CRS
            atol=cf.atol(),
            _test=True,  # _test=True to allow the dummy CRS
        )

        x = [99, 101, 199, 201, 299, 301, 399, 401, 499, 501, 599, 601]

        # Points through the W body
        index = locator.locate(y=150, x=x, latlon=False, unique=False)
        self.assertEqual(
            index.tolist(), [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2]
        )

        # Points through the U and W arms (a)
        index = locator.locate(y=201, x=x, latlon=False, unique=False)
        self.assertEqual(
            index.tolist(), [-2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -2]
        )

        # Points through the U and W arms (b)
        index = locator.locate(y=299, x=x, latlon=False, unique=False)
        self.assertEqual(
            index.tolist(), [-2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -2]
        )

        # Points through the U body
        index = locator.locate(y=350, x=x, latlon=False, unique=False)
        self.assertEqual(
            index.tolist(), [-2, -2, -2, 1, 1, 1, 1, 1, 1, -2, -2, -2]
        )

        # Points through neither U nor W (a)
        index = locator.locate(y=99, x=x, latlon=False, unique=False)
        self.assertEqual(index.tolist(), [-2] * len(x))

        # Points through neither U nor W (b)
        index = locator.locate(y=401, x=x, latlon=False, unique=False)
        self.assertEqual(index.tolist(), [-2] * len(x))

        # Points through neither U nor W (c)
        index = locator.locate(y=401, x=x, latlon=False)
        self.assertEqual(index.tolist(), [])

    def test_UGRID_locate_4(self):
        """Test cf.locate with a non-latitude_longitude mesh."""
        import pyproj

        from cf.mixin.utils.grid_mapping import create_projection_CRS

        f = cf.read("ugrid_global_1.nc")[0]

        # f0: latitude_longitude mesh
        f0 = f.copy()

        # Modify f to have a rotated_latitude_longitude mesh that is
        # geographically identical to f's latitude_longitude mesh.
        cr = cf.CoordinateReference()
        cr.set_coordinates({"auxiliarycoordinate1", "auxiliarycoordinate0"})
        cc = cr.coordinate_conversion
        cc.set_parameters(
            {
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_latitude": 38.0,
                "grid_north_pole_longitude": 190.0,
            }
        )

        proj_latlon = create_projection_CRS(cr, "latitude_longitude")
        proj_dst = create_projection_CRS(cr, "rotated_latitude_longitude")
        transformer = pyproj.Transformer.from_crs(
            proj_latlon, proj_dst, always_xy=True
        )

        lat = f.coord("Y").array
        lon = f.coord("X").array
        x, y = transformer.transform(lon, lat, errcheck=True, radians=False)
        f.coord("Y")[...] = y
        f.coord("X")[...] = x

        shape = f.coord("Y").bounds.shape
        lat = f.coord("Y").bounds.array.reshape(-1)
        lon = f.coord("X").bounds.array.reshape(-1)
        x, y = transformer.transform(lon, lat, errcheck=True, radians=False)
        f.coord("Y").bounds[...] = y.reshape(shape)
        f.coord("X").bounds[...] = x.reshape(shape)

        cr.clear_coordinates()
        cr.set_coordinates(["auxiliarycoordinate0", "auxiliarycoordinate1"])
        f.set_construct(cr)

        # Test that latlon=True locations give identical results for
        # both fields (latiude_longitude and
        # rotated_latitude_longitude)
        lat = f0.coord("Y").array
        lon = f0.coord("X").array

        i0 = cf.locate(lat, lon, f0)
        i = cf.locate(lat, lon, f)

        self.assertTrue((i0 == i).all())

        # Test that latlon=False locations for the
        # rotated_latitude_longitude gives identical results
        rlat = f.coord("Y").array
        rlon = f.coord("X").array

        j = cf.locate(rlat, rlon, f, latlon=False)

        self.assertTrue((i0 == j).all())

        # Now subspace the mesh
        N = 100
        index = slice(0, N)
        g = f.subspace(X=index)
        self.assertEqual(
            cf.locate(lat, lon, g).tolist(),
            cf.locate(lat, lon, f).tolist()[index],
        )

        result = cf.locate(lat, lon, g, unique=False).tolist()
        self.assertEqual(result.count(-g.size), f.size - N)

        result = cf.locate(rlat, rlon, g, latlon=False, unique=False).tolist()
        self.assertEqual(result.count(-g.size), f.size - N)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
