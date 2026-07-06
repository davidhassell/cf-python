import datetime
import unittest

import numpy as np

import cf


class LatLon2dTest(unittest.TestCase):
    """Test the creation of 2-d lat/lon coordinatesx."""

    def test_rotated_latitude_longitude(self):
        """Test rotated_latitude_longitude."""
        f = cf.read("rotated_pole.pp")[0]

        cr = f.coordinate_reference()
        self.assertEqual(
            cr.coordinate_conversion.parameters(),
            {
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_latitude": 38,
                "grid_north_pole_longitude": 190,
            },
        )

        self.assertFalse(f.auxiliary_coordinates())

        self.assertIsNone(f.create_latlon_coordinates(inplace=True))
        
        # Compare the 2-d lat/lon corodinates against
        # known-to-be-correct values
        lat = f.auxiliary_coordinate("latitude")
        self.assertEqual(lat.shape, (110, 106))
        self.assertTrue(np.allclose(lat[0, 0].array, 67.1246604))
        self.assertTrue(
            np.allclose(
                lat[0, 0].bounds.array,
                [67.13411912, 66.82618815, 67.11220769, 67.42286415],
            )
        )

        lon = f.auxiliary_coordinate("longitude")
        self.assertEqual(lon.shape, (110, 106))
        self.assertTrue(np.allclose(lon[0, 0].array, -45.98136153))
        self.assertTrue(
            np.allclose(
                lon[0, 0].bounds.array,
                [-46.7492162, -45.94548426, -45.21355527, -46.01992883],
            )
        )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
