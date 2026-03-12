# RIOMAR — HEALPix (DGGS) regridding

<!-- QUALITY_BADGE_START -->
[![Software quality](https://img.shields.io/badge/FAIRness-32%25-red "score: 32% | passed: 13 | failed: 27 | errors: 1")](RSFC_REPORT.md)
<!-- QUALITY_BADGE_END -->

This repository provides an end-to-end workflow to regrid ocean model data
(RiOMar/GAMAR) from curvilinear grids to
[HEALPix](https://healpix.jpl.nasa.gov/) (DGGS) format using xarray, Dask,
and Kerchunk. It runs both locally (HTTPS mode) and on HPC infrastructure.

1.  Define a Region Of Interest (ROI) from a lon/lat bounding box
2.  Prepare a temporary *small* Zarr dataset (for fast iteration and
    reproducible testing)
3.  Regrid variables to HEALPix using `healpix_regrid` (via
    `xarray.apply_ufunc`)
4.  Scale the same workflow to the full dataset on HPC and
    publish the resulting Zarr

The workflow assumes geographic coordinates in **EPSG:4326** and HEALPix
(WGS84) **nested indexing**.

------------------------------------------------------------------------

## Installation

```bash
# 1. Create the conda environment
conda env create -f notebook/environment.yml
conda activate riomar

# 2. Install the healpix_regrid package in editable mode
pip install -e ".[test]"
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Repository structure

```
healpix_regrid/         Reusable Python package (masking, kerchunk, dask, regridding)
tests/                  Pytest test suite for healpix_regrid
notebook/               Jupyter notebooks (interactive workflow & exploration)
bin/                    Python scripts for HPC batch runs
singularity_images/     Singularity container definitions
```

------------------------------------------------------------------------

## Workflow (recommended order)

### A. Create ROI from lon/lat bbox

**Notebook:** `Create_ROI_from_bbox.ipynb`

**Purpose** - Convert a lon/lat bounding box (EPSG:4326) into a HEALPix
(nested) ROI\
- Export parent-level cell IDs for fast indexing and masking\
- Export an outer boundary footprint (GeoJSON) for plotting and polygon
masking

**Input** - Bounding box `(min_lon, min_lat, max_lon, max_lat)` in
EPSG:4326\
- `child_level`\
- `parent_level`\
- `edge_level`

**Output** - HEALPix ROI cells at parent level: `parent_ids.npz`\
- Boundary footprint (for masking/subsetting before regridding):
`outer_boundary.geojson`

**Notes** - The notebook computes child-level cells covering the bbox,
then maps them to the parent level and builds polygons.

------------------------------------------------------------------------

### B. Prepare a temporary Zarr (fast iteration dataset)

**Notebook:** `Prep_regrid.ipynb`

**Purpose** - Open RiOMar data via a Kerchunk catalog (HPC filesystem or
HTTPS export)\
- Apply a spatial subset or mask using the ROI from step A\
- Write a lightweight Zarr locally (or on HPC scratch) to speed up
development

**Output** - A temporary Zarr dataset (e.g. `small.zarr`) used as input
for the regridding notebook

**Tip** - Set `OUT_ZARR` to an existing path on your machine or HPC
scratch.

------------------------------------------------------------------------

### C. Regrid to HEALPix using `apply_ufunc`

**Notebook:** `regrid_apply_ufunc.ipynb`

**Purpose** - Load the temporary Zarr created in **B**\
- Regrid selected variables to HEALPix using `healpix_regrid.to_healpix`\
- Align output to ROI-derived `cell_ids` (drop extra cells, fill missing
with `np.nan`)\
- Save a HEALPix-aligned, chunked Zarr

**Output** - A HEALPix-indexed dataset with a `cell_ids` coordinate
(nested indexing)

------------------------------------------------------------------------

## Scaling to HPC

On HPC (Datarmor), the `bin/` scripts run the same pipeline as the notebooks.
Scripts auto-detect the environment by checking whether the HPC filesystem
(`/scale/project/lops-oh-fair2adapt/`) exists.

```bash
# Submit a PBS job
qsub bin/submit.sh
```

Singularity container definitions are in `singularity_images/f2a_riomar/`
(layered build: hardened Debian base -> conda scientific stack -> JupyterHub).

------------------------------------------------------------------------

## Conventions / Expected Variables

-   Geographic coordinates: `nav_lon_*`, `nav_lat_*` in degrees\
-   HEALPix:
    -   Nested indexing\
    -   `cell_ids` coordinate\
    -   `level` attribute (e.g. 13)\
    -   WGS84 ellipsoid

------------------------------------------------------------------------

## Troubleshooting

-   **ROI boundary "edge" artifact**\
    Increase `edge_level` in `Create_ROI_from_bbox.ipynb`.

-   **Memory spikes**\
    Write intermediate results to Zarr and rechunk. Avoid premature
    `.compute()` calls.

-   **Mismatch between stacked dims and ufunc core dims**\
    Ensure `stack(point=(...))` matches `input_core_dims=[["point"]]`.

------------------------------------------------------------------------

## Extra notebooks

-   `Define_healpix_Parent_chunk.ipynb` —
    Experiments on parent levels together with bounding boxes.

-   `simple_regrid.ipynb` —
    Tests alternative regridding methods available in `regrid_to_healpix`.

-   `M1_*` / `M2_*` notebooks —
    Experiments with Kerchunk/Icechunk/VirtualiZarr creation and loading on Datarmor.

------------------------------------------------------------------------

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

This project is licensed under the [Apache License 2.0](LICENSE).