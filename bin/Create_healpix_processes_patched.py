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

import xdggs
import healpix_geo


def run():
    """Run the full pipeline.

    NOTE: With Dask LocalCluster(processes=True), worker processes may use the
    'spawn' start method, which re-imports this module. Therefore, cluster
    creation and all compute must be executed under the `if __name__ == "__main__"`
    guard to avoid recursive process spawning.
    """
    time_chunk_size = 24  # 1 day as a chunk
    #time_chunk_size = 1  # 1 day as a chunk

    child_level=13

    with np.load("parent_ids.npz") as data:
        parent_ids = data["parent_ids"]
        parent_level = int(data["parent_level"])



    HPC_PREFIX    = "/scale/project/lops-oh-fair2adapt/"
    HTTPS_PREFIX  = "https://data-fair2adapt.ifremer.fr/"
    CATALOG_PATH  = "fpaul/tmp/riomar_3months.json"
    #CATALOG_PATH  = "riomar-virtualizarr/Y2023.json"
    CATALOG_PATH  = "riomar-virtualizarr/YALL.json"
    OUT_PARQUET   = "riomar_3months_.parq"   # local parquet refs cache
    OUT_ZARR   = "riomar-zarr_tina/ALL2.zarr"  # local parquet refs cache


    def patch_kc_refs_inplace(kc, hpc_prefix=HPC_PREFIX, https_prefix=HTTPS_PREFIX):
        refs = kc.get("refs", kc.get("references"))
        if refs is None:
            raise KeyError("Can't find 'refs' (or 'references') in kerchunk JSON")

        def patch_target(x):
            if isinstance(x, str) and x.startswith(hpc_prefix):
                return https_prefix + x[len(hpc_prefix):]
            return x

        for k, v in list(refs.items()):
            if isinstance(v, list) and v and isinstance(v[0], str):
                refs[k] = [patch_target(v[0])] + v[1:]
            elif isinstance(v, str):
                refs[k] = patch_target(v)

        kc["refs"] = refs
        return kc


    # ------------------------------
    # 1) HPC mode: open directly
    # ------------------------------
    if Path(HPC_PREFIX).exists():
        ##on HPC


        zarr_hp_file_path = HPC_PREFIX + OUT_ZARR

        # pick a fast local path on the compute node
        local_dir = ( "/tmp")
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
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
            local_directory=local_dir,   # <--- THIS FIXES THE WARNING
            dashboard_address=":0",
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

        # Optional: per-worker details
        #for addr, w in workers.items():
         #   print(f"- {addr}: nthreads={w['nthreads']}, mem_limit={w['memory_limit']/1e9:.2f} GB »)
        KERCHUNK_CATALOG = HPC_PREFIX + CATALOG_PATH
        print("Running in HPC mode:", KERCHUNK_CATALOG)

        ds = xr.open_dataset(KERCHUNK_CATALOG, engine="kerchunk", chunks={})

    # ------------------------------
    # 2) HTTPS mode: prefer local parquet cache if present
    # ------------------------------
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        client
        zarr_hp_file_path =  OUT_ZARR

        KERCHUNK_CATALOG = HTTPS_PREFIX + CATALOG_PATH
        print("Running in HTTPS mode:", KERCHUNK_CATALOG)
        # If parquet refs already exist locally, open them (fast path)
        # This part is commented since on the fly transformation is faster than loading the parquet file in actual config
        # (check why at some point) 
        # Loading from local parquet is also slower than loading json and convert the path on the fly...
        # thus i deactivate the if here
        #if Path(OUT_PARQUET).exists():
        if False and Path(OUT_PARQUET).exists():
            print(f"✅ Found local parquet refs: ./{OUT_PARQUET} -> opening that")
            xr.open_dataset(OUT_PARQUET, engine="kerchunk", chunks={})

        # Else: fetch JSON, patch refs to https, open, AND write parquet refs cache
        else:
            print(f"ℹ️ No local parquet refs found at ./{OUT_PARQUET} -> creating them from JSON")

            with fsspec.open(KERCHUNK_CATALOG, "rt") as f:
                kc = json.load(f)

            kc = patch_kc_refs_inplace(kc)

            # open now (from in-memory dict)
            ds = xr.open_dataset(kc, engine="kerchunk", chunks={})

            ## write parquet refs cache for next time
            #import kerchunk.df as kcdf
            #kcdf.refs_to_dataframe(kc, OUT_PARQUET)
            #print("✅ Wrote kerchunk parquet refs to:", OUT_PARQUET)

    print(ds)
    ds=ds[['temp','salt','zeta']].assign_coords(
        nav_lon_rho=ds["nav_lon_rho"].load(),
        nav_lat_rho=ds["nav_lat_rho"].load(),
    )



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
    # ---------------------------------------------------------------------
    # HEALPix regridding (patched):
    # - Avoid stacking the full time dimension (build stack per time block)
    # - Avoid pandas MultiIndex / object coords by using create_index=False
    # - Build the bilinear operator ONCE from static lon/lat (time_counter=0)
    # - Write only the HEALPix-aligned dataset to Zarr in append mode
    # ---------------------------------------------------------------------

    import geopandas as gpd

    # 1) Read polygon (GeoJSON)
    gdf = gpd.read_file("outer_boundary.geojson")
    poly = gdf.geometry.iloc[0]

    # 2) Build a 2D ROI mask from zeta at time_counter=0 (polygon + not-null)
    zeta0 = ds.zeta.isel(time_counter=0).compute()
    zeta0_roi = apply_polygon_mask(zeta0, poly)   # returns DataArray masked outside polygon
    zeta_mask_2d = zeta0_roi.notnull()            # True where water points exist inside polygon

    # 3) Convert mask to a 1D point selector (integer positions), using stack without MultiIndex
    mask_1d = zeta_mask_2d.stack(point=("y_rho", "x_rho"), create_index=False)
    point_idx = np.where(mask_1d.values)[0]

    # 4) Build operator ONCE from static lon/lat at time_counter=0 for these points
    ds_geom = ds.isel(time_counter=0).stack(point=("y_rho", "x_rho"), create_index=False).isel(point=point_idx)
    lon = ds_geom["nav_lon_rho"].values.astype(np.float64)
    lat = ds_geom["nav_lat_rho"].values.astype(np.float64)

    from regrid_to_healpix.regrid_to_healpix_bilinear import Set
    nr = Set(lon_deg=lon, lat_deg=lat, level=child_level, device="cpu", threshold=0.5, ellipsoid="WGS84")
    cell_ids = np.asarray(nr.get_cell_ids(), dtype=np.int64)
    ncell = int(cell_ids.size)

    def to_healpix_point(data_1d):
        out = nr.transform(np.asarray(data_1d, dtype=np.float64), lam=0.1)
        return np.asarray(out, dtype=np.float64)

    # 5) Target cell ids for your final region-of-interest
    aligned_child_ids = np.unique(
        healpix_geo.nested.zoom_to(parent_ids, depth=parent_level, new_depth=child_level)
    )
    target_ids = aligned_child_ids.astype(cell_ids.dtype)

    # chunk sizes
    chunk_size = 4 ** (child_level - parent_level)
    block = time_chunk_size  # e.g., 24*100

    # choose variables to regrid (edit as needed)
    vars_to_regrid = [v for v in ["temp", "salt", "zeta"] if v in ds.data_vars]

    nt = ds.sizes["time_counter"]
    first = True

    for t0 in range(0, nt, block):
        t1 = min(nt, t0 + block)

        # Slice time FIRST, then stack just this time-block
        ds_blk = ds.isel(time_counter=slice(t0, t1))

        # Stack without MultiIndex + select the same valid points
        ds_in = ds_blk.stack(point=("y_rho", "x_rho"), create_index=False).isel(point=point_idx)

        # --- Regrid each variable (keeps graph smaller than applying to whole Dataset) ---
        out_vars = {}
        for v in vars_to_regrid:
            out_vars[v] = xr.apply_ufunc(
                to_healpix_point,
                ds_in[v],
                input_core_dims=[["point"]],
                output_core_dims=[["cell_ids"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={"output_sizes": {"cell_ids": ncell}},
            )

        ds_hp = xr.Dataset(out_vars).assign_coords(cell_ids=("cell_ids", cell_ids))

        # Re-attach cell_ids metadata
        ds_hp["cell_ids"].attrs.update({
            "grid_name": "healpix",
            "level": int(child_level),
            "indexing_scheme": "nested",
            "ellipsoid": "WGS84",
        })

        # Align to target ids and chunk for writing
        ds_aligned = (
            ds_hp.reindex(cell_ids=target_ids, fill_value=np.nan)
            .chunk({"time_counter": min(block, time_chunk_size), "cell_ids": chunk_size})
        )

        # Write / append along time
        if first:
            ds_aligned.to_zarr(zarr_hp_file_path, mode="w", consolidated=False, safe_chunks=False)
            first = False
        else:
            ds_aligned.to_zarr(
                zarr_hp_file_path,
                mode="a",
                append_dim="time_counter",
                consolidated=False,
                safe_chunks=False,
            )

    # consolidate once at the end (optional; makes opens faster)
    import zarr
    zarr.consolidate_metadata(zarr_hp_file_path)


if __name__ == '__main__':
    run()
