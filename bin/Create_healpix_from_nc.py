#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this takes 4 s
#
import json
import fsspec
from pathlib import Path
import os
import psutil
from dask.distributed import Client, LocalCluster
import xarray as xr
import numpy as np
import geopandas as gpd
import xdggs
import healpix_geo



def apply_polygon_mask(
    ds: xr.Dataset,
    poly,
    lon_name: str = "nav_lon_rho",
    lat_name: str = "nav_lat_rho",
    mask_name: str = "mask",
) -> xr.Dataset:
    """
    Add a boolean mask to ds that is True where (lon,lat) points fall inside `poly`.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain 2D lon/lat fields (e.g. nav_lon_rho, nav_lat_rho).
    poly : shapely.geometry (Polygon or MultiPolygon)
        In EPSG:4326 lon/lat.
    lon_name, lat_name : str
        Variable names for lon/lat inside ds.
    mask_name : str
        Name of the new coordinate/variable to store the mask.

    Returns
    -------
    xr.Dataset
        Same dataset with a new boolean DataArray `mask_name`
        attached as a coordinate (same dims as lon/lat).
    """
    lon2d = ds[lon_name]
    lat2d = ds[lat_name]

    # numpy arrays
    lon = lon2d.data if isinstance(lon2d.data, np.ndarray) else np.asarray(lon2d.values)
    lat = lat2d.data if isinstance(lat2d.data, np.ndarray) else np.asarray(lat2d.values)

    # --- build mask ---
    try:
        # shapely>=2
        from shapely import contains_xy
        mask_np = contains_xy(poly, lon, lat)
    except Exception:
        # fallback (fast-ish, but only uses exterior ring)
        from matplotlib.path import Path

        def _contains_one_polygon(p):
            x, y = p.exterior.xy
            path = Path(np.column_stack([x, y]))
            pts = np.column_stack([lon.ravel(), lat.ravel()])
            return path.contains_points(pts).reshape(lon.shape)

        # Support MultiPolygon by OR-ing components
        if getattr(poly, "geom_type", None) == "MultiPolygon":
            mask_np = np.zeros(lon.shape, dtype=bool)
            for p in poly.geoms:
                mask_np |= _contains_one_polygon(p)
        else:
            mask_np = _contains_one_polygon(poly)

    # wrap back to DataArray with same dims/coords as lon/lat
    mask_da = xr.DataArray(
        mask_np.astype(bool),
        coords=lon2d.coords,
        dims=lon2d.dims,
        name=mask_name,
    )

    # Attach mask (as a coord, like you were doing)
    return ds.assign_coords({mask_name: mask_da}).where(mask_da)#,drop=True)

# Build operator once

def to_healpix(ds_in):
    from regrid_to_healpix.regrid_to_healpix_bilinear import Set


    lon = ds_in["nav_lon_rho"].values.astype(np.float64)
    lat = ds_in["nav_lat_rho"].values.astype(np.float64)

    nr = Set(lon_deg=lon, lat_deg=lat, level=child_level, device="cpu", threshold=0.5, ellipsoid="WGS84")
    cell_ids = np.asarray(nr.get_cell_ids(), dtype=np.int64)
    ncell = int(cell_ids.size)

    def to_healpix_point(data_1d):
        out = nr.transform(np.asarray(data_1d, dtype=np.float64), lam=0.1)
        return np.asarray(out, dtype=np.float64)

    # Apply to the whole Dataset: only to chosen data_vars
    #vars_to_regrid = ["temp"]  # add "salt", "zeta", ...


    ds_hp = xr.apply_ufunc(
        to_healpix_point,
        ds_in,
        input_core_dims=[["point"]],
        output_core_dims=[["cell_ids"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"cell_ids": ncell}},
        keep_attrs=True,  # keep dataset + variable attrs where possible
    )

    # Re-attach coordinate + its metadata
    ds_hp = ds_hp.assign_coords(cell_ids=("cell_ids", cell_ids))
    ds_hp["cell_ids"].attrs.update({
        "grid_name": "healpix",
        "level": 13,
        "indexing_scheme": "nested",
        "ellipsoid": "WGS84",
    })


    # compute the child id from the final interest region
    aligned_child_ids = np.unique(healpix_geo.nested.zoom_to(
        parent_ids,
        depth=parent_level,
        new_depth=child_level
    ))

    # Make sure types match (important: your ds_hp cell_ids look like int64)
    target_ids = aligned_child_ids.astype(ds_hp["cell_ids"].dtype)
    #compute the chunk size 
    chunk_size=4**(child_level - parent_level )

    # aline the fill non existing values with np.nan, and take out non interestd zone
    #
    return  (
        ds_hp.reindex(cell_ids=target_ids, fill_value=np.nan)
        .chunk({"cell_ids": chunk_size},{"time_counter": time_chunk_size})
    )



time_chunk_size = 24  # 1 day as a chunk

child_level=13
nt = 201600

with np.load("parent_ids.npz") as data:
    parent_ids = data["parent_ids"]
    parent_level = int(data["parent_level"])



HPC_PREFIX    = "/scale/project/lops-oh-fair2adapt/"
HTTPS_PREFIX  = "https://data-fair2adapt.ifremer.fr/"
OUT_ZARR   = "riomar-zarr_tina/ALL.zarr"  # local parquet refs cache
FILE_PREFIX    = "/scale/project/lops-oh-fair2adapt/riomar/GAMAR/*Y"

# 1. Read the polygon from GeoJSON.

gdf=gpd.read_file("outer_boundary.geojson", driver="GeoJSON")
poly = gdf.geometry.iloc[0]  



# ------------------------------
# 1) HPC mode: open directly
# ------------------------------
if Path(HPC_PREFIX).exists():
    ##on HPC


    zarr_hp_file_path = HPC_PREFIX + OUT_ZARR

    # pick a fast local path on the compute node
    local_dir = (
     "/tmp"
    )
    local_dir = str(Path(local_dir) / "dask-scratch")
    print("Using Dask local_directory:", local_dir)


    print("=== Starting local Dask cluster (auto-sized) ===")

    cpu = os.cpu_count() or 1
    total_gb = psutil.virtual_memory().total / (1024**3)

    # Good “use most, but not all” defaults:
    n_workers = cpu                   # ~1 worker per CPU core
    threads_per_worker = 1            # best for numpy-heavy compute
    memory_limit_gb = (total_gb * 0.85) / n_workers  # leave ~15% headroom
    memory_limit = f"{memory_limit_gb:.2f}GB"
    n_workers=16
    cluster = LocalCluster(
        n_workers=n_workers,
    #    threads_per_worker=threads_per_worker,
        processes=True,
#        memory_limit=memory_limit,
        local_directory=local_dir,   # <--- THIS FIXES THE WARNING
#        dashboard_address=":0",
    )
    client = Client(cluster)

    print("Dask dashboard:", client.dashboard_link)

    print("\n=== Dask cluster resources ===")
    info = client.scheduler_info()
    workers = info["workers"]

    total_threads = sum(w["nthreads"] for w in workers.values())
    total_mem_gb = sum(w["memory_limit"] for w in workers.values()) / (1024**3)

    print(f"Workers: {len(workers)}")
    print(f"Total threads: {total_threads}")
    print(f"Total memory limit: {total_mem_gb:.2f} GB")



first = True


for Y in range(2001, 2024):
    print(Y)
    ds = xr.open_mfdataset(f"{FILE_PREFIX}{Y}*.nc")[['temp','salt','zeta']]
    ds = ds.assign_coords(
             nav_lon_rho=ds["nav_lon_rho"].load(),
             nav_lat_rho=ds["nav_lat_rho"].load(),
         ).chunk({"time_counter": time_chunk_size})
    print(ds)
    if first:
        #2. Trim the dataset  with polygon mask : zeta_mask
        #3. Find out which values are 'ground' by  computing not null values of zeta at time_counter=0 : zeta_mask
        zeta_mask= apply_polygon_mask(
        ds.zeta.isel(time_counter=0).compute(),
        poly).notnull().compute()
        #zeta_mask
    print(zeta_mask)
    ds_roi = ds
    ds_roi['zeta_mask']=zeta_mask
        
    #4. Stack the spatial coordinate and drop all the ground point (ds_in)
    
    ds_roi_1d=ds_roi.stack(point=("y_rho", "x_rho") )
    
    ds_roi_1d = ds_roi_1d.where(ds_roi_1d.zeta_mask,drop=True).drop_vars('zeta_mask').persist()
    print(ds_roi_1d)

    ds_in  = to_healpix(ds_roi_1d)
    

    if first:
        
        ds_in.to_zarr(zarr_hp_file_path, mode="w", consolidated=False, safe_chunks=False)
        first = False
    else:
        ds_in.to_zarr(
            zarr_hp_file_path,
            mode="a",
            append_dim="time_counter",
            consolidated=False,
            safe_chunks=False,
        )

# consolidate once (optional)
import zarr
zarr.consolidate_metadata(zarr_hp_file_path)


