# RIOMAR → HEALPix (DGGS) regridding notebooks

This folder contains the end-to-end workflow to:

1.  Define a Region Of Interest (ROI) from a lon/lat bounding box\
2.  Prepare a temporary *small* Zarr dataset (for fast iteration and
    reproducible testing)\
3.  Regrid variables to HEALPix using `regrid_to_healpix` (via
    `xarray.apply_ufunc`)\
4.  (Later) Scale the same workflow to the full dataset on HPC and
    publish the resulting Zarr

The workflow assumes geographic coordinates in **EPSG:4326** and HEALPix
(WGS84) **nested indexing**.

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
- Build the HEALPix operator once (from `nav_lon_*` / `nav_lat_*`)\
- Regrid selected variables with `xarray.apply_ufunc`\
- Align output to ROI-derived `cell_ids` (drop extra cells, fill missing
with `np.nan`)\
- Save a HEALPix-aligned, chunked Zarr


**Output** - A HEALPix-indexed dataset with a `cell_ids` coordinate
(nested indexing)

**TODO** - align may be better to be taken care of in the function used in apply_u_func..

------------------------------------------------------------------------

## Scaling to HPC ("big scale" run)

(Documentation to be added.)

Planned publication target (once validated):

    https://data-fair2adapt.ifremer.fr/riomar-zarr/small_hp.zarr

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

-   `Define_healpix_parent_chunk.ipynb`\
    Experiments on parent levels together with bounding boxes.

-   `simple_regrid.ipynb`\
    Tests alternative regridding methods available in
    `regrid_to_healpix`.

-   Notebooks experimenting with Kerchunk creation and loading on
    Datarmor\
    (to be added by Fred)
