import datetime
import unittest

import numpy as np

import cf

healpix_imported = True
try:
    import healpix  # noqa: F401
except ImportError:
    healpix_imported = False


class DataTest(unittest.TestCase):
    """Unit tests for HEALPix utilities."""

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Skip all if healpix module not available!
        if not healpix_imported:
            self.skipTest(
                "Test module requires 'healpix' package. Install it to run all."
            )

        # Create matching lists of selected nested, ring, nuniq and zuniq
        # indices for every refinement level.
        indices = [
            (r, i, healpix.nest2ring(healpix.order2nside(r), i))
            for r in range(30)
            for i in (0, 7, (12 * 4**r) - 1)
        ]
        self.refinement_levels, self.nested_indices, self.ring_indices = map(
            list, zip(*indices)
        )

        self.nuniq_indices = [  # noqa: F841
            i + 4 ** (1 + r)
            for r, i in zip(self.refinement_levels, self.nested_indices)
        ]

        self.zuniq_indices = [  # noqa: F841
            (2 * i + 1) * 4 ** (29 - r)
            for r, i in zip(self.refinement_levels, self.nested_indices)
        ]

    def test_HEALPix_uniq2zuniq(self):
        """Test _uniq2zuniq"""
        from cf.data.dask_utils_healpix import _uniq2zuniq

        self.assertTrue(
            np.array_equal(_uniq2zuniq(self.nuniq_indices), self.zuniq_indices)
        )

    def test_HEALPix_zuniq2uniq(self):
        """Test _zuniq2uniq"""
        from cf.data.dask_utils_healpix import _zuniq2uniq

        self.assertTrue(
            np.array_equal(_zuniq2uniq(self.zuniq_indices), self.nuniq_indices)
        )

    def test_HEALPix_zuniq2pix(self):
        """Test _zuniq2pix"""
        from cf.data.dask_utils_healpix import _zuniq2pix

        # nested
        order, i = _zuniq2pix(self.zuniq_indices, nest=True)

        self.assertTrue(np.array_equal(order, self.refinement_levels))
        self.assertTrue(np.array_equal(i, self.nested_indices))

        # ring
        with self.assertRaises(NotImplementedError):
            _zuniq2pix(self.zuniq_indices, nest=False)

    def test_HEALPix_pix2zuniq(self):
        """Test _pix2zuniq"""
        from cf.data.dask_utils_healpix import _pix2zuniq

        # nested
        z = [
            _pix2zuniq(r, i, nest=True)
            for r, i in zip(self.refinement_levels, self.nested_indices)
        ]

        self.assertTrue(np.array_equal(z, self.zuniq_indices))

        # ring
        z = [
            _pix2zuniq(r, i, nest=False)
            for r, i in zip(self.refinement_levels, self.ring_indices)
        ]

        self.assertTrue(np.array_equal(z, self.zuniq_indices))

    def test_HEALPix_locate_1(self):
        """Test cf.locate with a HEALPix grid."""
        # ------------------------------------------------------------
        # nested/ring/nuniq/zuniq with one order
        # ------------------------------------------------------------
        f = cf.example_field(12)

        # Get the location for every HEALPix cell
        g = f.create_latlon_coordinates()
        y = g.coordinate("Y").array[::-1]
        x = g.coordinate("X").array[::-1]

        for scheme in ("nested", "ring", "nuniq", "zuniq"):
            g = g.healpix_change_indexing_scheme(scheme)
            self.assertEqual(cf.locate(y, x, g).tolist(), list(range(y.size)))

        for scheme in ("nested", "ring"):
            f = f.healpix_change_indexing_scheme(scheme)
            self.assertEqual(
                cf.locate(y, x, f, unique=False).tolist(),
                list(range(y.size - 1, -1, -1)),
            )

        for scheme in ("nested", "ring"):
            f = f.healpix_change_indexing_scheme(scheme)
            self.assertEqual(
                cf.locate(y, x, f[:, :15], unique=True).tolist(),
                list(range(15)),
            )

        for scheme in ("nested", "ring"):
            f = f.healpix_change_indexing_scheme(scheme)
            self.assertEqual(
                cf.locate(y, x, f[:, :15], unique=False).tolist(),
                [-15] * (y.size - 15) + list(range(15 - 1, -1, -1)),
            )

        for scheme in ("nested", "ring"):
            f = f.healpix_change_indexing_scheme(scheme)
            self.assertEqual(
                cf.locate([-78, 34, 34], [200, 48, 48], f).tolist(),
                [0, 40],
            )

        for scheme in ("nested", "ring"):
            f = f.healpix_change_indexing_scheme(scheme)
            self.assertEqual(
                cf.locate(
                    [-78, 34, 34], [200, 48, 48], f, unique=False
                ).tolist(),
                [40, 0, 0],
            )

        # unique=False not implemented for nuniq/zuniq
        for scheme in ("nuniq", "zuniq"):
            g = g.healpix_change_indexing_scheme(scheme)
            with self.assertRaises(NotImplementedError):
                cf.locate(y, x, g, unique=False)

        # ------------------------------------------------------------
        # nuniq/zuniq with multiple orders
        # ------------------------------------------------------------
        f = cf.example_field(13)

        # Get the location for every HEALPix cell
        g = f.create_latlon_coordinates()
        y = g.coordinate("Y").array[::-1]
        x = g.coordinate("X").array[::-1]

        for scheme in ("nuniq", "zuniq"):
            g = g.healpix_change_indexing_scheme(scheme)
            self.assertEqual(
                cf.locate(y, x, g).tolist(),
                list(range(y.size)),
            )

        i = 40
        for scheme in ("nuniq", "zuniq"):
            g = g.healpix_change_indexing_scheme(scheme)
            self.assertEqual(
                cf.locate(y, x, g[:, -i:]).tolist(),
                list(range(i)),
            )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
