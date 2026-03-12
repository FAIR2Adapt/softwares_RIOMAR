import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from healpix_regrid.masking import _build_mask, apply_polygon_mask


def _make_grid(nx=10, ny=10, lon_range=(-5, 5), lat_range=(45, 55)):
    """Create a simple xarray Dataset with 2-D lon/lat on a regular grid."""
    lon_1d = np.linspace(lon_range[0], lon_range[1], nx)
    lat_1d = np.linspace(lat_range[0], lat_range[1], ny)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    ds = xr.Dataset(
        {"temp": (("y_rho", "x_rho"), np.random.rand(ny, nx))},
        coords={
            "nav_lon_rho": (("y_rho", "x_rho"), lon2d),
            "nav_lat_rho": (("y_rho", "x_rho"), lat2d),
        },
    )
    return ds


class TestBuildMask:
    def test_points_inside_polygon(self):
        poly = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        lon = np.array([0.0, 2.0, -0.5])
        lat = np.array([0.0, 2.0, 0.5])

        mask = _build_mask(poly, lon, lat)

        assert mask[0] is np.True_  # inside
        assert mask[1] is np.False_  # outside
        assert mask[2] is np.True_  # inside

    def test_2d_arrays(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        lon = np.array([[1, 5, 15], [2, 8, 20]])
        lat = np.array([[1, 5, 5], [2, 8, 5]])

        mask = _build_mask(poly, lon, lat)

        assert mask.shape == (2, 3)
        assert mask[0, 0]  # inside
        assert mask[0, 1]  # inside
        assert not mask[0, 2]  # outside
        assert not mask[1, 2]  # outside

    def test_multipolygon(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        multi = MultiPolygon([p1, p2])

        lon = np.array([0.5, 5.5, 3.0])
        lat = np.array([0.5, 5.5, 3.0])

        mask = _build_mask(multi, lon, lat)

        assert mask[0]  # in p1
        assert mask[1]  # in p2
        assert not mask[2]  # outside both


class TestApplyPolygonMask:
    def test_masks_dataset(self):
        ds = _make_grid()
        # Polygon covering roughly the center of the grid
        poly = Polygon([(-2, 48), (2, 48), (2, 52), (-2, 52)])

        result = apply_polygon_mask(ds, poly)

        assert "mask" in result.coords
        # Points outside polygon should be NaN
        assert result["temp"].isnull().any()
        # Points inside polygon should have values
        assert result["temp"].notnull().any()

    def test_works_with_dataarray(self):
        ds = _make_grid()
        poly = Polygon([(-2, 48), (2, 48), (2, 52), (-2, 52)])

        result = apply_polygon_mask(ds["temp"], poly)

        assert "mask" in result.coords
        assert result.isnull().any()
        assert result.notnull().any()

    def test_custom_coord_names(self):
        lon2d, lat2d = np.meshgrid(np.arange(5), np.arange(5))
        ds = xr.Dataset(
            {"data": (("y", "x"), np.ones((5, 5)))},
            coords={
                "lon": (("y", "x"), lon2d.astype(float)),
                "lat": (("y", "x"), lat2d.astype(float)),
            },
        )
        poly = Polygon([(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)])

        result = apply_polygon_mask(ds, poly, lon_name="lon", lat_name="lat")

        assert "mask" in result.coords

    def test_all_outside_gives_all_nan(self):
        ds = _make_grid(lon_range=(100, 110), lat_range=(0, 10))
        poly = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])

        result = apply_polygon_mask(ds, poly)

        assert result["temp"].isnull().all()
