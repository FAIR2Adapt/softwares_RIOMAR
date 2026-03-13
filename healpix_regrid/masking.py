from __future__ import annotations

import numpy as np
import xarray as xr


def apply_polygon_mask(
    ds: xr.Dataset | xr.DataArray,
    poly,
    lon_name: str = "nav_lon_rho",
    lat_name: str = "nav_lat_rho",
    mask_name: str = "mask",
) -> xr.Dataset | xr.DataArray:
    """Add a boolean mask that is True where (lon, lat) points fall inside *poly*.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Must contain 2-D lon/lat fields.
    poly : shapely.geometry.Polygon or MultiPolygon
        Region of interest in EPSG:4326.
    lon_name, lat_name : str
        Variable names for longitude / latitude inside *ds*.
    mask_name : str
        Name of the new boolean coordinate.

    Returns
    -------
    Same type as *ds*, with a boolean coordinate *mask_name* attached and
    values outside the polygon set to NaN.
    """
    lon2d = ds[lon_name]
    lat2d = ds[lat_name]

    lon = np.asarray(lon2d.values)
    lat = np.asarray(lat2d.values)

    mask_np = _build_mask(poly, lon, lat)

    mask_da = xr.DataArray(
        mask_np,
        coords=lon2d.coords,
        dims=lon2d.dims,
        name=mask_name,
    )

    return ds.assign_coords({mask_name: mask_da}).where(mask_da)


def _build_mask(poly, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return a boolean array — True where (lon, lat) is inside *poly*."""
    try:
        from shapely import contains_xy

        return contains_xy(poly, lon, lat).astype(bool)
    except ImportError:
        pass

    # Fallback for shapely < 2
    from matplotlib.path import Path as MplPath

    def _contains_one(p):
        x, y = p.exterior.xy
        path = MplPath(np.column_stack([x, y]))
        pts = np.column_stack([lon.ravel(), lat.ravel()])
        return path.contains_points(pts).reshape(lon.shape)

    if getattr(poly, "geom_type", None) == "MultiPolygon":
        mask = np.zeros(lon.shape, dtype=bool)
        for p in poly.geoms:
            mask |= _contains_one(p)
        return mask

    return _contains_one(poly)
