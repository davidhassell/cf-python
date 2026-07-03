import datetime
import unittest

import numpy as np

import cf


class LatLon2dTest(unittest.TestCase):
    """Test the creation of 2-d lat/lon."""

    def test_rotated_latitude_longitude_0(self):
        """Test rotated_latitude_longitude."""
        # Test round trip
        import pyproj

        from cf.mixin.utils.grid_mapping import (
            latitude_longitude,
            rotated_latitude_longitude,
        )

        cr = cf.CoordinateReference()
        cc = cr.coordinate_conversion
        cc.set_parameter("grid_mapping_name", "rotated_latitude_longitude")
        cc.set_parameter("grid_north_pole_latitude", 38.0)
        cc.set_parameter("grid_north_pole_longitude", 190.0)

        proj_src = rotated_latitude_longitude(cr)
        proj_latlon = latitude_longitude(None)

        transformer0 = pyproj.Transformer.from_crs(
            proj_latlon, proj_src, always_xy=True
        ).transform
        transformer1 = pyproj.Transformer.from_crs(
            proj_src, proj_latlon, always_xy=True
        ).transform

        # Centres
        lon0 = np.array([[1, 1]], float)
        lat0 = np.array([[50, 60]], float)
        gridx , gridy = transformer0(lon0, lat0)
        lon1, lat1 = transformer1(gridx , gridy )

        self.assertTrue(np.allclose(lon0, lon1))
        self.assertTrue(np.allclose(lat0, lat1))
swap 1 and 0
        # Bounds
        blon0 = np.array([[0, 0, 2, 2], [0, 0, 2, 2]], float)
        blat0 = np.array([[51, 49, 49, 51], [61, 59, 59, 61]], float)
        bgridx , bgridy = transformer0(blon0, blat0)
        blon1, blat1 = transformer1( bgridx , bgridy)

        self.assertTrue(np.allclose(blon0, blon1))
        self.assertTrue(np.allclose(blat0, blat1))
        print()
        print('gridx=', gridx, 'bgridx=', bgridx)
        print('gridy=', gridy, 'bgridy=', bgridy)

        # Test with Field
        f = cf.example_field(0)
        f = f[:2, 0]

        key_x, x = f.dimension_coordinate("X", item=True)
        x.data[...] = gridx[0, 0]
        print('x.array=', x.array)

        x.bounds.data[...] = bgridx[0, [0, -1]]
        x.override_units("degrees", inplace=True)
        x.standard_name = "grid_longitude"

        key_y, y = f.dimension_coordinate("Y", item=True)
        y.data[...] = gridy
        print('y.array=', y.array)
        y.bounds.data[0] = bgridy[0, [0, 1]]
        y.bounds.data[1] = bgridy[1, [0, 1]]
        y.override_units("degrees", inplace=True)
        y.standard_name = "grid_latitude"

        fcr = cf.CoordinateReference()
        fcr.coordinate_conversion.set_parameter(
            "grid_mapping_name", "rotated_latitude_longitude"
        )
        fcr.coordinate_conversion.set_parameter(
            "grid_north_pole_latitude", 38.0
        )
        fcr.coordinate_conversion.set_parameter(
            "grid_north_pole_longitude", 190.0
        )
        f.set_construct(fcr, copy=False)

        self.assertEqual(len(f.auxiliary_coordinates()), 0)

        for coordinates in (set(), {key_x, key_y}):
            fcr.clear_coordinates()
            fcr.set_coordinates(coordinates)

            g = f.create_latlon_coordinates()

            self.assertEqual(len(g.auxiliary_coordinates()), 2)

            gcr = g.coordinate_reference()
            self.assertEqual(
                gcr.coordinates(),
                {key_x, key_y, "auxiliarycoordinate0", "auxiliarycoordinate1"},
            )

            lat = g.auxiliary_coordinate('latitude')
            print ()
            print(lat.array, lat0)
            self.assertTrue(np.allclose(lat.array, lat0))
            
            lon = g.auxiliary_coordinate('longitude')
            self.assertTrue(np.allclose(lon.array, lon0))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
