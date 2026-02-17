#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create_healpix.py

Auto-generated from Create_healpix.ipynb, then cleaned for readability.
Edits applied:
- Removed Jupyter cell markers
- Normalized whitespace (max 2 blank lines)
- Stripped trailing spaces
"""

#!/usr/bin/env python

# #  regridding open RiOMar/GAMAR via Kerchunk → write a HEALPIX Zarr
#
# This notebook make create HEALPIxZARR of ROI>
#
# It opens the dataset from a **Kerchunk catalog** generated on HPC, and the *same* workflow can open:
# - directly on the HPC filesystem, or
# - over HTTPS via the DATARMOR export service (`https://data-fair2adapt.ifremer.fr/`).
#
# ## Goal
# Produce a temporary Zarr (e.g. `small.zarr`) that is then used as input to:
# - `regrid_apply_ufunc.ipynb`
#
# ## Outputs
# - `OUT_ZARR`: a lightweight subset written to Zarr (local or HPC scratch)
#
# ## Steps
# 1. **Open the dataset** from a Kerchunk catalog (portable: HPC filesystem or HTTPS export)
# 2. **Select variables** and ensure lon/lat are explicit coordinates
# 3. **Build a polygon mask** helper for the model grid
# 4. **Define the ROI** and apply the mask (load ROI polygon, mask, and subset)
# 5. **regrid**
# 6. **Write a  Zarr**
#
#
#

# ## Kerchunk catalog options
#
# A **Kerchunk catalog** is a JSON mapping that lets Xarray open a collection of NetCDF (or similar) files as a *virtual* Zarr dataset.
# Depending on where you run this notebook, you can point to:
#
# - **HPC filesystem path** (fast, when you have direct access to `/scale/...`)
# - **HTTPS export** (portable, when you access the same data through `https://data-fair2adapt.ifremer.fr/`)
#
# Below are example catalog paths that have been created previously (kept here as a reference).
#

# ## 1. **Open the dataset** from a Kerchunk catalog (portable: HPC filesystem or HTTPS export)
#
# Many Kerchunk catalogs store references to the original files (often as absolute HPC paths).
# When opening through HTTPS, we **rewrite** those references from:
#
# `/scale/project/lops-oh-fair2adapt/...` → `https://data-fair2adapt.ifremer.fr/...`
#
# The cell below:
# 1. Detects whether the HPC path exists (so we are running on the cluster).
# 2. Otherwise loads the JSON over HTTPS, patches references in-memory, and opens the dataset.
# 3. (Optional) can cache the patched references locally as a parquet file for faster re-opening.
#


get_ipython().run_cell_magic('time', '', '# this takes 4 s\n#\nimport json\nimport fsspec\nfrom pathlib import Path\nimport os\nimport psutil\nfrom dask.distributed import Client, LocalCluster\nimport xarray as xr\nimport numpy as np\n\nimport xdggs\nimport healpix_geo\n\ntime_chunk_size = 24  # 1 day as a chunk\nchild_level=13\n\nwith np.load("parent_ids.npz") as data:\n    parent_ids = data["parent_ids"]\n    parent_level = int(data["parent_level"])\n\n\n\nHPC_PREFIX    = "/scale/project/lops-oh-fair2adapt/"\nHTTPS_PREFIX  = "https://data-fair2adapt.ifremer.fr/"\nCATALOG_PATH  = "fpaul/tmp/riomar_3months.json"\n#CATALOG_PATH  = "riomar-virtualizarr/Y2023.json"\nCATALOG_PATH  = "riomar-virtualizarr/YALL.json"\nOUT_PARQUET   = "riomar_3months_.parq"   # local parquet refs cache\nOUT_ZARR   = "riomar-zarr_tina/Y2023.zarr"  # local parquet refs cache\n\n\ndef patch_kc_refs_inplace(kc, hpc_prefix=HPC_PREFIX, https_prefix=HTTPS_PREFIX):\n    refs = kc.get("refs", kc.get("references"))\n    if refs is None:\n        raise KeyError("Can\'t find \'refs\' (or \'references\') in kerchunk JSON")\n\n    def patch_target(x):\n        if isinstance(x, str) and x.startswith(hpc_prefix):\n            return https_prefix + x[len(hpc_prefix):]\n        return x\n\n    for k, v in list(refs.items()):\n        if isinstance(v, list) and v and isinstance(v[0], str):\n            refs[k] = [patch_target(v[0])] + v[1:]\n        elif isinstance(v, str):\n            refs[k] = patch_target(v)\n\n    kc["refs"] = refs\n    return kc\n\n\n# ------------------------------\n# 1) HPC mode: open directly\n# ------------------------------\nif Path(HPC_PREFIX).exists():\n    ##on HPC\n\n\n    zarr_hp_file_path = HPC_PREFIX + OUT_ZARR\n\n    # pick a fast local path on the compute node\n    local_dir = (\n     "/tmp"\n    )\n    local_dir = str(Path(local_dir) / "dask-scratch")\n    print("Using Dask local_directory:", local_dir)\n\n\n    print("=== Starting local Dask cluster (auto-sized) ===")\n\n    cpu = os.cpu_count() or 1\n    total_gb = psutil.virtual_memory().total / (1024**3)\n\n    # Good “use most, but not all” defaults:\n    n_workers = cpu                   # ~1 worker per CPU core\n    threads_per_worker = 1            # best for numpy-heavy compute\n    memory_limit_gb = (total_gb * 0.85) / n_workers  # leave ~15% headroom\n    memory_limit = f"{memory_limit_gb:.2f}GB"\n    n_workers=32\n    cluster = LocalCluster(\n        n_workers=n_workers,\n    #    threads_per_worker=threads_per_worker,\n        processes=True,\n        memory_limit=memory_limit,\n        local_directory=local_dir,   # <--- THIS FIXES THE WARNING\n        dashboard_address=":8787",\n    )\n    client = Client(cluster)\n\n    print("Dask dashboard:", client.dashboard_link)\n\n    print("\\n=== Dask cluster resources ===")\n    info = client.scheduler_info()\n    workers = info["workers"]\n\n    total_threads = sum(w["nthreads"] for w in workers.values())\n    total_mem_gb = sum(w["memory_limit"] for w in workers.values()) / (1024**3)\n\n    print(f"Workers: {len(workers)}")\n    print(f"Total threads: {total_threads}")\n    print(f"Total memory limit: {total_mem_gb:.2f} GB")\n\n    # Optional: per-worker details\n    #for addr, w in workers.items():\n     #   print(f"- {addr}: nthreads={w[\'nthreads\']}, mem_limit={w[\'memory_limit\']/1e9:.2f} GB »)\n    KERCHUNK_CATALOG = HPC_PREFIX + CATALOG_PATH\n    print("Running in HPC mode:", KERCHUNK_CATALOG)\n\n    ds = xr.open_dataset(KERCHUNK_CATALOG, engine="kerchunk", chunks={})\n\n# ------------------------------\n# 2) HTTPS mode: prefer local parquet cache if present\n# ------------------------------\nelse:\n    cluster = LocalCluster()\n    client = Client(cluster)\n    client\n    zarr_hp_file_path =  OUT_ZARR\n\n    KERCHUNK_CATALOG = HTTPS_PREFIX + CATALOG_PATH\n    print("Running in HTTPS mode:", KERCHUNK_CATALOG)\n    # If parquet refs already exist locally, open them (fast path)\n    # This part is commented since on the fly transformation is faster than loading the parquet file in actual config\n    # (check why at some point) \n    # Loading from local parquet is also slower than loading json and convert the path on the fly...\n    # thus i deactivate the if here\n    #if Path(OUT_PARQUET).exists():\n    if False and Path(OUT_PARQUET).exists():\n        print(f"✅ Found local parquet refs: ./{OUT_PARQUET} -> opening that")\n        xr.open_dataset(OUT_PARQUET, engine="kerchunk", chunks={})\n\n    # Else: fetch JSON, patch refs to https, open, AND write parquet refs cache\n    else:\n        print(f"ℹ️ No local parquet refs found at ./{OUT_PARQUET} -> creating them from JSON")\n\n        with fsspec.open(KERCHUNK_CATALOG, "rt") as f:\n            kc = json.load(f)\n\n        kc = patch_kc_refs_inplace(kc)\n\n        # open now (from in-memory dict)\n        ds = xr.open_dataset(kc, engine="kerchunk", chunks={})\n\n        ## write parquet refs cache for next time\n        #import kerchunk.df as kcdf\n        #kcdf.refs_to_dataframe(kc, OUT_PARQUET)\n        #print("✅ Wrote kerchunk parquet refs to:", OUT_PARQUET)\n\nds\n')


# ## 2. **Select variables** and ensure lon/lat are explicit coordinates
#
# The original dataset contains many variables. For this demo we keep:
#
# - `temp` (temperature)
# - `salt` (salinity)
# - `zeta` (sea surface height)
#
# We also **load** the 2D longitude/latitude fields and attach them as coordinates (`nav_lon_rho`, `nav_lat_rho`).
# Loading them explicitly avoids repeated remote reads later (plots, masking, regridding, etc.).
#


ds=ds[['temp','salt','zeta']].assign_coords(
    nav_lon_rho=ds["nav_lon_rho"].load(),
    nav_lat_rho=ds["nav_lat_rho"].load(),
)
ds


# ## 3. **Build a polygon mask** helper for the model grid
# To extract a spatial subset, we load a boundary polygon (GeoJSON) and create a boolean mask on the dataset grid:
#
# - `True`  → grid point is inside the polygon
# - `False` → outside
#
# This is useful to reduce the dataset to a Region Of Interest (ROI) before saving or regridding.
#


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
# ----------------------------------------------------------------------------
# Build HEALPix regridding operator
# ----------------------------------------------------------------------------


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
    ds_aligned = (
        ds_hp.reindex(cell_ids=target_ids, fill_value=np.nan)
        .chunk({"cell_ids": chunk_size},{"time_counter": time_chunk_size})
    )
    return ds_aligned



# ## 4. **Define the ROI** and apply the mask (load ROI polygon, mask, and subset)
#
# For the regridding demo we focus on a limited area defined by an **outer boundary** polygon (stored in `outer_boundary.geojson`).
# We will:
#
# 1. Read the polygon from GeoJSON.
# 2. Trim the dataset  with polygon mask
# 3. Find out which values are 'ground' by  computing not null values of zeta at time_counter=0
# 4. Stack the spatial coordinate and drop all the ground point
#


import geopandas as gpd
# 1. Read the polygon from GeoJSON.

gdf=gpd.read_file("outer_boundary.geojson", driver="GeoJSON")
poly = gdf.geometry.iloc[0]  
#2. Trim the dataset  with polygon mask : zeta_mask
#3. Find out which values are 'ground' by  computing not null values of zeta at time_counter=0 : zeta_mask

zeta_mask= apply_polygon_mask(
    ds.zeta.isel(time_counter=0).compute(),
    poly).notnull()
#zeta_mask
ds_roi = ds
ds_roi['zeta_mask']=zeta_mask

#4. Stack the spatial coordinate and drop all the ground point (ds_in)

ds_roi_1d=ds_roi.stack(point=("y_rho", "x_rho") )

ds_roi_1d = ds_roi_1d.where(ds_roi_1d.zeta_mask,drop=True).drop_vars('zeta_mask')
ds_roi_1d


# ## 5.convert to healpix
#
#
#


ds_roi_1d = ds_roi_1d.chunk({"time_counter": time_chunk_size})
#ds_in = ds_roi_1d.isel(time_counter= slice(0,24*100))

block = 24   # 48 (or 24*100 etc.)
nt = ds_roi_1d.sizes["time_counter"]
nt = 48 
first = True
for t0 in range(0, nt, block):
    t1 = min(nt, t0 + block)

    ds_in  = to_healpix(ds_roi_1d.isel(time_counter=slice(t0, t1)))

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
zarr.convenience.consolidate_metadata(zarr_hp_file_path)

