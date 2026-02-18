from __future__ import annotations

from functools import partial
from pathlib import Path
import glob
import os

import psutil
import xarray as xr
from dask.distributed import Client, LocalCluster

from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
HPC_PREFIX = Path("/scale/project/lops-oh-fair2adapt/")
BASE_PATH = HPC_PREFIX / "riomar" / "GAMAR"
CATALOG_PATH = HPC_PREFIX / "riomar-virtualizarr2"
YEARS = range(2001, 2024)

LOADABLE_VARS = [
    "time_counter",
    "time_instant",
    "x_rho",
    "y_rho",
    "x_u",
    "x_v",
    "y_u",
    "y_v",
    "axis_nbounds",
]


# -----------------------------------------------------------------------------
# Dask setup
# -----------------------------------------------------------------------------
def start_client() -> Client:
    if HPC_PREFIX.exists():
        local_dir = Path("/tmp") / "dask-scratch"
        local_dir.mkdir(parents=True, exist_ok=True)

        cpu = os.cpu_count() or 1
        total_gb = psutil.virtual_memory().total / (1024**3)

        # Reasonable defaults; tune if needed
        n_workers = min(cpu, 32)              # cap at 32 like you intended
        threads_per_worker = 1
        memory_limit_gb = (total_gb * 0.85) / n_workers
        memory_limit = f"{memory_limit_gb:.2f}GB"

        print("Using Dask local_directory:", str(local_dir))
        print("=== Starting local Dask cluster ===")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=memory_limit,
            local_directory=str(local_dir),
            dashboard_address=":8787",  # if port busy, Dask will auto-pick another
        )
    else:
        print("HPC prefix not found; starting default LocalCluster()")
        cluster = LocalCluster()

    client = Client(cluster)

    print("Dask dashboard:", client.dashboard_link)
    info = client.scheduler_info()
    workers = info["workers"]
    total_threads = sum(w["nthreads"] for w in workers.values())
    total_mem_gb = sum(w["memory_limit"] for w in workers.values()) / (1024**3)

    print("\n=== Dask cluster resources ===")
    print(f"Workers: {len(workers)}")
    print(f"Total threads: {total_threads}")
    print(f"Total memory limit: {total_mem_gb:.2f} GB\n")
    return client


# -----------------------------------------------------------------------------
# Virtualizarr setup
# -----------------------------------------------------------------------------
def make_registry(base_path: Path) -> ObjectStoreRegistry:
    local_store = LocalStore()
    return ObjectStoreRegistry({f"file://{base_path}": local_store})


def main() -> None:
    client = start_client()

    parser = HDFParser()
    registry = make_registry(BASE_PATH)

    dask_open_vds = partial(
        open_virtual_dataset,
        loadable_variables=LOADABLE_VARS,
        parser=parser,
        registry=registry,
        decode_times=True,
    )

    CATALOG_PATH.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        print(year, "year")

        pattern = str(BASE_PATH / f"GAMAR_1h_inst_Y{year}*.nc")
        #files = glob.glob(pattern)
        files = sorted(glob.glob(pattern))
        print("files:",files,  len(files))

        if not files:
            print("No files found for", year, "- skipping\n")
            continue

        # Open datasets on workers, gather to client
        futures = client.map(dask_open_vds, files)
        dss = client.gather(futures)

        # Concat into one virtual dataset
        dss = sorted(dss, key=lambda _ds: _ds["time_counter"].values[0])
        ds = (
            xr.concat(
                dss,
                dim="time_counter",
                compat="override",
                coords="minimal",
                combine_attrs="drop_conflicts",
            )
            .set_coords(["time_counter_bounds", "time_instant_bounds"])
        )
        #ds = ds.sortby("time_counter")

        print(ds)

        outfile = CATALOG_PATH / f"Y{year}.json"
        ds.virtualize.to_kerchunk(str(outfile), format="json")
        print("Wrote:", outfile, "\n")


if __name__ == "__main__":
    main()
