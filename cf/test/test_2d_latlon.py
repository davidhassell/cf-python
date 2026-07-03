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
            proj_src, proj_latlon, always_xy=True
        ).transform
        transformer1 = pyproj.Transformer.from_crs(
            proj_latlon, proj_src, always_xy=True
        ).transform
        print()
        # Centres
        x0 = np.array([-10, 5], float)
        y0 = np.array([-10, 0, 20], float)  
        lon, lat = transformer0(*np.meshgrid(x0, y0))
        x1, y1 = transformer1(lon, lat)
        print(x1)
        print(y1)
        self.assertTrue(np.allclose(x1, x1[0]))
        x1 = x1[0]
        self.assertTrue(np.allclose(y1, y1[:, [0]]))
        y1 = y1[:,0]        
        self.assertTrue(np.allclose(x0,x1))
        self.assertTrue(np.allclose(y0,y1))

        # Bounds
        bx0 = np.array([[-20, 0], [0, 10]], float)
        by0 = np.array([[-15, -5], [-5, 5], [15, 25]], float)
        lon_bnds_2d = np.broadcast_to(bx0[np.newaxis, :, :], (3, 2, 2))
        lat_bnds_2d = np.broadcast_to(by0[:, np.newaxis, :], (3, 2, 2))

        full_lon_bnds = np.zeros((3, 2, 4))
        full_lat_bnds = np.zeros((3, 2, 4))
        
        # Corner 0: Bottom-Left  (min lat, min lon)
        full_lon_bnds[..., 0] = lon_bnds_2d[..., 0]
        full_lat_bnds[..., 0] = lat_bnds_2d[..., 0]
        
        # Corner 1: Top-Left     (max lat, min lon)
        full_lon_bnds[..., 1] = lon_bnds_2d[..., 0]
        full_lat_bnds[..., 1] = lat_bnds_2d[..., 1]
        
        # Corner 2: Top-Right    (max lat, max lon)
        full_lon_bnds[..., 2] = lon_bnds_2d[..., 1]
        full_lat_bnds[..., 2] = lat_bnds_2d[..., 1]
        
        # Corner 3: Bottom-Right (min lat, max lon)
        full_lon_bnds[..., 3] = lon_bnds_2d[..., 1]
        full_lat_bnds[..., 3] = lat_bnds_2d[..., 0]
        print(full_lon_bnds)
        print(full_lat_bnds)

        blon, blat = transformer0(full_lon_bnds, full_lat_bnds)
        bx1, by1 = transformer1( blon, blat)
        print(blon)
        print(blat)
        self.assertTrue(np.allclose(bx1, full_lon_bnds))
        self.assertTrue(np.allclose(by1, full_lat_bnds))

        # Test with Field
        f = cf.example_field(0)
        f = f[:3, :2]

        key_x, x = f.dimension_coordinate("X", item=True)
        x.data[...] = x0

        x.bounds.data[...] = bx0
        x.override_units("degrees", inplace=True)
        x.standard_name = "grid_longitude"

        key_y, y = f.dimension_coordinate("Y", item=True)
        y.data[...] = y0
        y.bounds.data[...] = by0
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
            self.assertTrue(np.allclose(lat.array, lat))
            print('----------')
            print(lat.bounds.array)
            print(blat)
            print(lat.bounds.array-blat)
            self.assertTrue(np.allclose(lat.bounds.array, blat))
            
            lon = g.auxiliary_coordinate('longitude')
            self.assertTrue(np.allclose(lon.array, lon))
            self.assertTrue(np.allclose(lon.bounds.array, blon))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
