import datetime
import unittest

import numpy as np
import pyproj

import cf

ellps = "WGS84"
km = "km"

f0 = cf.example_field(0)[0, 0]

key_x, x = f0.dimension_coordinate("X", item=True)
x.del_bounds()
x.standard_name = "projection_x_coordinate"
x.override_units(km, inplace=True)

key_y, y = f0.dimension_coordinate("Y", item=True)
y.del_bounds()
y.standard_name = "projection_y_coordinate"
y.override_units(km, inplace=True)

cr = cf.CoordinateReference()
cr.datum.set_parameters({"reference_ellipsoid_name": ellps})
cr.set_coordinates((key_x, key_y))
f0.set_construct(cr)

paris_lon = 2.2945  # La Tour Eiffel, WGS84
paris_lat = 48.8584  # La Tour Eiffel, WGS84

longlat = pyproj.CRS.from_string("+proj=longlat +ellps=WGS84")


def check_paris(g, atol=1e13, verbose=False):
    """Check if field `g` has Paris coordinates."""
    if verbose:
        print(
            [
                g.auxiliary_coordinate("X").array,
                g.auxiliary_coordinate("Y").array,
            ],
            [paris_lon, paris_lat],
        )

    ok = np.allclose(g.auxiliary_coordinate("X"), paris_lon, rtol=0, atol=atol)
    ok = ok & np.allclose(
        g.auxiliary_coordinate("Y"), paris_lat, rtol=0, atol=atol
    )
    return ok


def set_coordinate_conversion(f, parameters):
    """Replace the coordinate reference of field `f` with new parameters."""
    cr = f.coordinate_reference()

    cr.coordinate_conversion.clear_parameters()
    cr.coordinate_conversion.set_parameters(parameters)


def field_paris(proj):
    """Return a field for Paris with projection grid `proj`."""
    t = pyproj.Transformer.from_crs(longlat, proj, always_xy=1)
    x_coords, y_coords = t.transform(paris_lon, paris_lat)

    f = f0.copy()
    x = f.dimension_coordinate("X")
    x[...] = x_coords

    y = f.dimension_coordinate("Y")
    y[...] = y_coords

    rotated_latitude_longitude = proj.to_dict().get("proj") == "ob_tran"
    if rotated_latitude_longitude:
        x.standard_name = "grid_longitude"
        y.standard_name = "grid_latitude"
        x.override_units("degrees", inplace=True)
        y.override_units("degrees", inplace=True)

    return f


class LatLon2dTest(unittest.TestCase):
    """Test the creation of 2-d lat/lon coordinates."""

    def test_Field_2d_create_latlon_coordinates_albers_equal_area(self):
        """Test albers_equal_area."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_1 = 43
        lat_2 = 62
        lat_0 = 30
        lon_0 = 10
        proj = pyproj.CRS(
            proj="aea",
            lat_1=lat_1,
            lat_2=lat_2,
            lat_0=lat_0,
            lon_0=lon_0,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "albers_equal_area",
                "longitude_of_central_meridian": lon_0,
                "latitude_of_projection_origin": lat_0,
                "standard_parallel": [lat_1, lat_2],
            },
        )

        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "albers_equal_area",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_azimuthal_equidistant(self):
        """Test azimuthal_equidistant."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_0 = 48.8584
        lon_0 = 2.2945
        proj = pyproj.CRS(
            proj="aeqd",
            lat_0=lat_0,
            lon_0=lon_0,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "azimuthal_equidistant",
                "longitude_of_projection_origin": lon_0,
                "latitude_of_projection_origin": lat_0,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "azimuthal_equidistant",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_geostationary(self):
        """Test geostationary."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        h = 35785831
        lon_0 = 0
        sweep = "y"
        proj = pyproj.CRS(
            proj="geos",
            lon_0=lon_0,
            h=h,
            x_0=0,
            y_0=0,
            sweep=sweep,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "geostationary",
                "longitude_of_projection_origin": lon_0,
                "latitude_of_projection_origin": 0,
                "perspective_point_height": h,
                "sweep_angle_axis": sweep,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g, atol=1e-12))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f, {"grid_mapping_name": "geostationary", "crs_wkt": proj.to_wkt()}
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_lambert_azimuthal_equal_area(
        self,
    ):
        """Test lambert_azimuthal_equal_area."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_0 = 52
        lon_0 = 10
        x_0 = 4321000
        y_0 = 3210000
        proj = pyproj.CRS(
            proj="laea",
            lon_0=lon_0,
            lat_0=lat_0,
            x_0=x_0,
            y_0=y_0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "lambert_azimuthal_equal_area",
                "longitude_of_projection_origin": lon_0,
                "latitude_of_projection_origin": lat_0,
                "false_easting": x_0,
                "false_northing": y_0,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g, atol=1e-8))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "lambert_azimuthal_equal_area",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_lambert_conformal_conic(self):
        """Test lambert_conformal_conic."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_1 = 33
        lat_2 = 45
        lat_0 = 39
        lon_0 = -96
        proj = pyproj.CRS(
            proj="lcc",
            lon_0=lon_0,
            lat_0=lat_0,
            lat_1=lat_1,
            lat_2=lat_2,
            ellps=ellps,
            x_0=0,
            y_0=0,
            units=km,
        )
        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "lambert_conformal_conic",
                "longitude_of_central_meridian": lon_0,
                "latitude_of_projection_origin": lat_0,
                "standard_parallel": [lat_1, lat_2],
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "lambert_conformal_conic",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_lambert_cylindrical_equal_area(
        self,
    ):
        """Test lambert_cylindrical_equal_area."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lon_0 = 0
        lat_ts = 30
        proj = pyproj.CRS(
            proj="cea",
            lon_0=lon_0,
            lat_ts=lat_ts,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "lambert_cylindrical_equal_area",
                "longitude_of_central_meridian": lon_0,
                "standard_parallel": lat_ts,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "lambert_cylindrical_equal_area",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_mercator(self):
        """Test mercator."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lon_0 = 0
        lat_ts = 0
        proj = pyproj.CRS(
            proj="merc",
            lon_0=lon_0,
            lat_ts=lat_ts,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "mercator",
                "longitude_of_projection_origin": lon_0,
                "standard_parallel": lat_ts,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f, {"grid_mapping_name": "mercator", "crs_wkt": proj.to_wkt()}
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_oblique_mercator(self):
        """Test oblique_mercator."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_0 = 45
        lonc = 10
        alpha = 45
        k_0 = 1
        proj = pyproj.CRS(
            proj="omerc",
            lonc=lonc,
            lat_0=lat_0,
            alpha=alpha,
            k_0=k_0,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "oblique_mercator",
                "azimuth_of_central_line": alpha,
                "latitude_of_projection_origin": lat_0,
                "longitude_of_projection_origin": lonc,
                "scale_factor_at_projection_origin": k_0,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "oblique_mercator",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_orthographic(self):
        """Test orthographic."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_0 = 48.8584
        lon_0 = 2.2945
        proj = pyproj.CRS(
            proj="ortho",
            lon_0=lon_0,
            lat_0=lat_0,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "orthographic",
                "longitude_of_projection_origin": lon_0,
                "latitude_of_projection_origin": lat_0,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f, {"grid_mapping_name": "orthographic", "crs_wkt": proj.to_wkt()}
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_polar_stereographic(self):
        """Test polar_stereographic."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_ts = 90
        lat_0 = 90
        lon_0 = 0
        proj = pyproj.CRS(
            proj="stere",
            lon_0=lon_0,
            lat_0=lat_0,
            lat_ts=lat_ts,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "polar_stereographic",
                "longitude_of_projection_origin": lon_0,
                "latitude_of_projection_origin": lat_0,
                "standard_parallel": lat_ts,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "polar_stereographic",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_rotated_latitude_longitude(
        self,
    ):
        """Test rotated_latitude_longitude."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lon_0 = 190
        o_lat_p = 38
        o_lon_p = 0
        proj = pyproj.CRS(
            proj="ob_tran",
            o_proj="longlat",
            o_lon_p=o_lon_p,
            o_lat_p=o_lat_p,
            lon_0=lon_0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_latitude": o_lat_p,
                "grid_north_pole_longitude": lon_0,
                "north_pole_grid_longitude": o_lon_p,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "rotated_latitude_longitude",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_sinusoidal(self):
        """Test sinusoidal."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lon_0 = 0
        proj = pyproj.CRS(
            proj="sinu",
            lon_0=lon_0,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "sinusoidal",
                "longitude_of_projection_origin": lon_0,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f, {"grid_mapping_name": "sinusoidal", "crs_wkt": proj.to_wkt()}
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_stereographic(self):
        """Test stereographic."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_0 = 90
        lon_0 = 0
        k_0 = 0.994
        proj = pyproj.CRS(
            proj="stere",
            lon_0=lon_0,
            lat_0=lat_0,
            k_0=k_0,
            x_0=0,
            y_0=0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "stereographic",
                "longitude_of_projection_origin": lon_0,
                "latitude_of_projection_origin": lat_0,
                "scale_factor_at_projection_origin": k_0,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f, {"grid_mapping_name": "stereographic", "crs_wkt": proj.to_wkt()}
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_transverse_mercator(self):
        """Test transverse_mercator."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        lat_0 = 0
        lon_0 = 3
        k_0 = 0.9996012717
        x_0 = 500000
        y_0 = 0
        proj = pyproj.CRS(
            proj="tmerc",
            lon_0=lon_0,
            lat_0=lat_0,
            k_0=k_0,
            x_0=x_0,
            y_0=y_0,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "transverse_mercator",
                "longitude_of_central_meridian": lon_0,
                "latitude_of_projection_origin": lat_0,
                "scale_factor_at_central_meridian": k_0,
                "false_easting": x_0,
                "false_northing": y_0,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "transverse_mercator",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_vertical_perspective(self):
        """Test vertical_perspective."""
        # Get the correct projected coordinates for Paris, convert
        # these projected coordinates to lat/lon using
        # `cf.Field.create_latlon_coordinates`, and check that we end
        # up in Paris.
        h = 3000000
        lat_0 = 48.8584
        lon_0 = 2.2945
        proj = pyproj.CRS(
            proj="nsper",
            lon_0=lon_0,
            lat_0=lat_0,
            h=h,
            ellps=ellps,
            units=km,
        )

        f = field_paris(proj)
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "vertical_perspective",
                "longitude_of_projection_origin": lon_0,
                "latitude_of_projection_origin": lat_0,
                "perspective_point_height": h,
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

        # Do the same, but with a WKT-defined coordinate reference
        # construct.
        set_coordinate_conversion(
            f,
            {
                "grid_mapping_name": "vertical_perspective",
                "crs_wkt": proj.to_wkt(),
            },
        )
        g = f.create_latlon_coordinates()
        self.assertTrue(check_paris(g))

    def test_Field_2d_create_latlon_coordinates_bounds(self):
        """Test lat/lon bounds."""
        # Check that lat/lon coordinate bounds are correctly created.
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

        # Compare the 2-d lat/lon coordinates against
        # known-to-be-correct values
        lat = f.auxiliary_coordinate("latitude")
        self.assertEqual(lat.shape, (110, 106))
        self.assertTrue(np.allclose(lat[0, 0].array, 67.1246604))

        self.assertEqual(lat.bounds.units, lat.units)
        self.assertTrue(
            np.allclose(
                lat[0, 0].bounds.array,
                [67.13411912, 66.82618815, 67.11220769, 67.42286415],
            )
        )

        lon = f.auxiliary_coordinate("longitude")
        self.assertEqual(lon.shape, (110, 106))
        self.assertTrue(np.allclose(lon[0, 0].array, -45.98136153))

        self.assertEqual(lon.bounds.units, lon.units)
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
