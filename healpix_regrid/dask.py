from __future__ import annotations

import os
from pathlib import Path

from dask.distributed import Client, LocalCluster


def start_client(
    hpc_prefix: str = "/scale/project/lops-oh-fair2adapt/",
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    memory_fraction: float = 0.85,
    local_directory: str | None = None,
) -> Client:
    """Create a Dask ``LocalCluster`` and return a connected ``Client``.

    When *hpc_prefix* exists on the filesystem the cluster is tuned for
    HPC nodes (scratch directory, memory caps).  Otherwise a lightweight
    default cluster is started.

    Parameters
    ----------
    hpc_prefix : str
        Path whose existence signals an HPC environment.
    n_workers : int or None
        Number of workers.  ``None`` → ``min(cpu_count, 32)`` on HPC,
        Dask default otherwise.
    threads_per_worker : int
        Threads per worker (1 is best for NumPy-heavy workloads).
    memory_fraction : float
        Fraction of total RAM to allocate across all workers on HPC.
    local_directory : str or None
        Dask scratch directory (HPC only, defaults to ``/tmp/dask-scratch``).
    """
    import psutil

    if Path(hpc_prefix).exists():
        local_dir = Path(local_directory or "/tmp/dask-scratch")
        local_dir.mkdir(parents=True, exist_ok=True)

        cpu = os.cpu_count() or 1
        total_gb = psutil.virtual_memory().total / (1024**3)
        nw = n_workers or min(cpu, 32)
        mem_limit = f"{(total_gb * memory_fraction) / nw:.2f}GB"

        cluster = LocalCluster(
            n_workers=nw,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=mem_limit,
            local_directory=str(local_dir),
            dashboard_address=":0",
        )
    else:
        cluster = LocalCluster(
            **({"n_workers": n_workers} if n_workers else {}),
        )

    client = Client(cluster)

    info = client.scheduler_info()
    workers = info["workers"]
    total_threads = sum(w["nthreads"] for w in workers.values())
    total_mem_gb = sum(w["memory_limit"] for w in workers.values()) / (1024**3)

    print(f"Dask dashboard: {client.dashboard_link}")
    print(
        f"Workers: {len(workers)}  |  Threads: {total_threads}"
        f"  |  Memory: {total_mem_gb:.1f} GB"
    )

    return client
