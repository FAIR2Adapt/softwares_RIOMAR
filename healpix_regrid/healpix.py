from __future__ import annotations

import healpix_geo
import numpy as np
import xarray as xr


def to_healpix(
    ds_in: xr.Dataset,
    parent_ids: np.ndarray,
    parent_level: int,
    child_level: int = 13,
    time_chunk_size: int = 24,
    point_dim: str = "point",
    lon_name: str = "nav_lon_rho",
    lat_name: str = "nav_lat_rho",
    device: str = "cpu",
    lam: float = 0.1,
) -> xr.Dataset:
    """Regrid a stacked xarray Dataset to HEALPix using bilinear interpolation.

    Parameters
    ----------
    ds_in : xr.Dataset
        Input dataset stacked along *point_dim* (e.g. via
        ``ds.stack(point=("y_rho", "x_rho"))``).
    parent_ids : np.ndarray
        Parent-level HEALPix cell IDs defining the region of interest.
    parent_level : int
        HEALPix depth of *parent_ids*.
    child_level : int
        Target HEALPix resolution level.
    time_chunk_size : int
        Chunk size along the time dimension for the output.
    point_dim : str
        Name of the spatial dimension in the stacked input.
    lon_name, lat_name : str
        Coordinate names for longitude / latitude.
    device : str
        ``"cpu"`` or ``"cuda"`` (passed to ``regrid_to_healpix``).
    lam : float
        Smoothing parameter for the bilinear transform.

    Returns
    -------
    xr.Dataset
        HEALPix-indexed dataset aligned to the ROI cell IDs, chunked and
        ready for Zarr output.
    """
    from regrid_to_healpix.regrid_to_healpix_bilinear import Set

    lon = ds_in[lon_name].values.astype(np.float64)
    lat = ds_in[lat_name].values.astype(np.float64)

    nr = Set(
        lon_deg=lon,
        lat_deg=lat,
        level=child_level,
        device=device,
        threshold=0.5,
        ellipsoid="WGS84",
    )
    cell_ids = np.asarray(nr.get_cell_ids(), dtype=np.int64)
    ncell = int(cell_ids.size)

    def _transform(data_1d):
        out = nr.transform(np.asarray(data_1d, dtype=np.float64), lam=lam)
        return np.asarray(out, dtype=np.float64)

    ds_hp = xr.apply_ufunc(
        _transform,
        ds_in,
        input_core_dims=[[point_dim]],
        output_core_dims=[["cell_ids"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"cell_ids": ncell}},
        keep_attrs=True,
    )

    ds_hp = ds_hp.assign_coords(cell_ids=("cell_ids", cell_ids))
    ds_hp["cell_ids"].attrs.update(
        {
            "grid_name": "healpix",
            "level": int(child_level),
            "indexing_scheme": "nested",
            "ellipsoid": "WGS84",
        }
    )

    # Align to the target ROI cell IDs
    aligned_child_ids = np.unique(
        healpix_geo.nested.zoom_to(
            parent_ids, depth=parent_level, new_depth=child_level
        )
    )
    target_ids = aligned_child_ids.astype(ds_hp["cell_ids"].dtype)
    chunk_size = 4 ** (child_level - parent_level)

    return ds_hp.reindex(cell_ids=target_ids, fill_value=np.nan).chunk(
        {"cell_ids": chunk_size, "time_counter": time_chunk_size}
    )


def write_zarr_chunked(
    ds: xr.Dataset,
    zarr_path: str,
    regrid_fn,
    time_dim: str = "time_counter",
    block_size: int | None = None,
) -> None:
    """Write a dataset to Zarr in time blocks, regridding each block.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (stacked, masked, chunked).
    zarr_path : str
        Output Zarr store path.
    regrid_fn : callable
        Function that takes a time-sliced dataset and returns a
        HEALPix-regridded dataset (e.g. a partial of :func:`to_healpix`).
    time_dim : str
        Name of the time dimension.
    block_size : int or None
        Number of time steps per block.  ``None`` → process all at once.
    """
    import zarr

    nt = ds.sizes[time_dim]
    block = block_size or nt
    first = True

    for t0 in range(0, nt, block):
        t1 = min(nt, t0 + block)
        ds_block = regrid_fn(ds.isel({time_dim: slice(t0, t1)}))

        if first:
            ds_block.to_zarr(zarr_path, mode="w", consolidated=False, safe_chunks=False)
            first = False
        else:
            ds_block.to_zarr(
                zarr_path,
                mode="a",
                append_dim=time_dim,
                consolidated=False,
                safe_chunks=False,
            )

    zarr.consolidate_metadata(zarr_path)
