"""HEALPix grid metadata for RiOMar outputs.

One place that builds the grid metadata written by the regridding scripts, so
that every output describes its grid in the same way:

* the Zarr DGGS convention v1 (``zarr_conventions`` + ``dggs`` group
  attributes), https://github.com/zarr-conventions/dggs/blob/v1/README.md
* the CF 1.13 ``healpix`` grid mapping (grid mapping variable, the
  ``healpix_index`` standard name on the cell coordinate and ``grid_mapping``
  on the data variables),
  https://cfconventions.org/cf-conventions/release/v1.13.0/cf-conventions.html#healpix

``healpix_nside`` / ``healpix_order`` are kept on the grid mapping variable,
with the same values as the standard attributes, because the RiOMar dashboard
reads them.  The older per-coordinate attributes (``grid_name`` / ``level`` /
``indexing_scheme`` / ``ellipsoid`` on the cell coordinate) are removed
instead: next to the CF ``standard_name`` they make ``xdggs.decode`` (0.7)
pick its old convention and fail, while the ``dggs`` group attribute carries
the same information.

The attribute builders and :func:`check_healpix_metadata` only use the Python
standard library, so they can be used to patch or check stores on machines
without the scientific stack (see ``bin/fix_healpix_zarr_metadata.py``).
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

DGGS_CONVENTION: dict[str, str] = {
    "schema_url": (
        "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json"
    ),
    "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38",
    "name": "dggs",
    "description": "Discrete Global Grid Systems convention for zarr",
}
"""Convention declaration of the Zarr DGGS convention v1."""

# WGS84 (EPSG:7030 ellipsoid, EPSG:6326 datum, EPSG:4326 geographic CRS).
WGS84 = {
    "name": "WGS84",
    "semi_major_axis": 6378137.0,
    "inverse_flattening": 298.257223563,
    # CF names, see CF 1.13 Appendix F, table F.1
    "reference_ellipsoid_name": "WGS 84",
    "horizontal_datum_name": "WGS_1984",
    "prime_meridian_name": "Greenwich",
    "geographic_crs_name": "WGS 84",
}

ELLIPSOIDS = {"WGS84": WGS84}

INDEXING_SCHEMES = ("nested", "ring")
LEGACY_CELL_KEYS = ("grid_name", "level", "indexing_scheme", "ellipsoid", "nest")
"""Old per-coordinate grid attributes, superseded by the ``dggs`` attribute."""
MAX_REFINEMENT_LEVEL = 29


def _check_grid(level: int, indexing_scheme: str) -> None:
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"refinement level must be an int, got {level!r}")
    if not 0 <= level <= MAX_REFINEMENT_LEVEL:
        raise ValueError(
            f"refinement level must be in [0, {MAX_REFINEMENT_LEVEL}], got {level}"
        )
    if indexing_scheme not in INDEXING_SCHEMES:
        raise ValueError(
            f"indexing_scheme must be one of {INDEXING_SCHEMES}, "
            f"got {indexing_scheme!r}"
        )


def _ellipsoid(name: str) -> dict[str, Any]:
    try:
        return ELLIPSOIDS[name.upper()]
    except KeyError:
        raise ValueError(
            f"unknown ellipsoid {name!r}, known: {sorted(ELLIPSOIDS)}"
        ) from None


def dggs_attrs(
    level: int,
    spatial_dimension: str,
    coordinate: str,
    indexing_scheme: str = "nested",
    ellipsoid: str = "WGS84",
) -> dict[str, Any]:
    """``dggs`` object of the Zarr DGGS convention v1 for a HEALPix group."""
    _check_grid(level, indexing_scheme)
    ell = _ellipsoid(ellipsoid)
    return {
        "name": "healpix",
        "refinement_level": int(level),
        "indexing_scheme": indexing_scheme,
        "spatial_dimension": spatial_dimension,
        "coordinate": coordinate,
        "compression": "none",
        "ellipsoid": {
            "name": ell["name"],
            "semi_major_axis": ell["semi_major_axis"],
            "inverse_flattening": ell["inverse_flattening"],
        },
    }


def grid_mapping_attrs(
    level: int,
    indexing_scheme: str = "nested",
    ellipsoid: str = "WGS84",
    legacy: bool = True,
) -> dict[str, Any]:
    """Attributes of the CF 1.13 ``healpix`` grid mapping variable."""
    _check_grid(level, indexing_scheme)
    ell = _ellipsoid(ellipsoid)
    attrs = {
        "grid_mapping_name": "healpix",
        "indexing_scheme": indexing_scheme,
        "refinement_level": int(level),
        "semi_major_axis": ell["semi_major_axis"],
        "inverse_flattening": ell["inverse_flattening"],
        "longitude_of_prime_meridian": 0.0,
        # CF: these four names must be all defined if any one is defined
        "reference_ellipsoid_name": ell["reference_ellipsoid_name"],
        "horizontal_datum_name": ell["horizontal_datum_name"],
        "prime_meridian_name": ell["prime_meridian_name"],
        "geographic_crs_name": ell["geographic_crs_name"],
        "long_name": f"HEALPix grid on the {ell['name']} ellipsoid",
    }
    if legacy:
        attrs["healpix_nside"] = 2**level
        attrs["healpix_order"] = indexing_scheme
    return attrs


def cell_coordinate_attrs(
    level: int,
    indexing_scheme: str = "nested",
) -> dict[str, Any]:
    """Attributes of the coordinate holding the HEALPix cell indices."""
    _check_grid(level, indexing_scheme)
    return {
        "standard_name": "healpix_index",
        "long_name": (
            f"HEALPix cell index ({indexing_scheme} scheme, refinement level {level})"
        ),
    }


def with_dggs_convention(conventions: Iterable[Mapping[str, Any]] | None) -> list:
    """Return ``zarr_conventions`` with the DGGS v1 declaration present once."""
    out = [dict(c) for c in (conventions or [])]
    out = [
        c
        for c in out
        if c.get("uuid") != DGGS_CONVENTION["uuid"]
        and c.get("schema_url") != DGGS_CONVENTION["schema_url"]
        and c.get("spec_url") != DGGS_CONVENTION["spec_url"]
    ]
    out.append(dict(DGGS_CONVENTION))
    return out


def group_attrs(
    attrs: Mapping[str, Any],
    level: int,
    spatial_dimension: str,
    coordinate: str,
    indexing_scheme: str = "nested",
    ellipsoid: str = "WGS84",
    history: str | None = None,
) -> dict[str, Any]:
    """Group attributes with the DGGS convention, from existing ``attrs``.

    A ``Conventions`` attribute inherited from the model output (for example
    ``"CF-1.6"`` written by XIOS) is removed: the regridded data uses a grid
    mapping that only exists since CF 1.13, so the old declaration is wrong,
    and the whole dataset has not been checked against CF 1.13.  The removal
    is recorded in ``history``.
    """
    out = copy.deepcopy(dict(attrs))
    notes = []
    source_conventions = out.pop("Conventions", None)
    if source_conventions is not None:
        notes.append(
            f"source attribute Conventions={source_conventions!r} removed: the "
            "HEALPix grid metadata follows the CF 1.13 healpix grid mapping and "
            "the Zarr DGGS convention v1; the rest of the dataset was not "
            "checked against CF 1.13"
        )
    if history:
        notes.insert(0, history)
    if notes:
        previous = out.get("history")
        out["history"] = "\n".join(([previous] if previous else []) + notes)
    out["zarr_conventions"] = with_dggs_convention(out.get("zarr_conventions"))
    out["dggs"] = dggs_attrs(
        level,
        spatial_dimension=spatial_dimension,
        coordinate=coordinate,
        indexing_scheme=indexing_scheme,
        ellipsoid=ellipsoid,
    )
    return out


def apply_healpix_metadata(
    ds,
    level: int,
    cell_coord: str = "cell_ids",
    crs_name: str = "crs",
    indexing_scheme: str = "nested",
    ellipsoid: str = "WGS84",
    legacy: bool = True,
    history: str | None = None,
):
    """Write the HEALPix grid metadata on an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a 1-D integer coordinate ``cell_coord`` holding the
        HEALPix cell indices.
    level : int
        HEALPix refinement level of ``cell_coord``.
    cell_coord : str
        Name of the cell index coordinate.  Its (single) dimension is the
        spatial dimension of the DGGS.
    crs_name : str
        Name of the scalar grid mapping variable that is added.
    indexing_scheme : {"nested", "ring"}
        HEALPix indexing scheme of the cell indices.
    ellipsoid : str
        Reference ellipsoid; only ``"WGS84"`` is defined.
    legacy : bool
        Also write ``healpix_nside`` / ``healpix_order`` on the grid mapping
        variable (read by the RiOMar dashboard).
    history : str, optional
        Line appended to the ``history`` group attribute.

    Returns
    -------
    xr.Dataset
        A new dataset; the data arrays are not copied or computed.
    """
    import numpy as np

    if cell_coord not in ds.variables:
        raise KeyError(f"cell coordinate {cell_coord!r} not found")
    cell = ds[cell_coord]
    if cell.ndim != 1:
        raise ValueError(f"{cell_coord!r} must be 1-D, got dims {cell.dims}")
    if not np.issubdtype(cell.dtype, np.integer):
        raise TypeError(f"{cell_coord!r} must hold integers, got {cell.dtype}")
    (dim,) = cell.dims

    out = ds.copy(deep=False)
    out[cell_coord].attrs = {
        k: v for k, v in out[cell_coord].attrs.items() if k not in LEGACY_CELL_KEYS
    } | cell_coordinate_attrs(level, indexing_scheme)

    out[crs_name] = (
        (),
        np.int8(0),
        grid_mapping_attrs(level, indexing_scheme, ellipsoid, legacy=legacy),
    )

    for name, var in out.data_vars.items():
        if name != crs_name and dim in var.dims:
            var.attrs["grid_mapping"] = crs_name
            if cell_coord != dim:
                coords = var.attrs.get("coordinates", "").split()
                if cell_coord not in coords:
                    var.attrs["coordinates"] = " ".join(coords + [cell_coord])

    out.attrs = group_attrs(
        out.attrs,
        level,
        spatial_dimension=dim,
        coordinate=cell_coord,
        indexing_scheme=indexing_scheme,
        ellipsoid=ellipsoid,
        history=history,
    )
    return out


def check_healpix_metadata(
    attrs: Mapping[str, Any],
    variables: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Check the HEALPix grid metadata of one group.

    Parameters
    ----------
    attrs : mapping
        Group attributes.
    variables : mapping
        ``{name: {"dims": [...], "shape": [...], "attrs": {...}}}`` for every
        array of the group.

    Returns
    -------
    list of str
        Problems found; empty if the metadata is consistent.
    """
    problems = []
    conventions = attrs.get("zarr_conventions") or []
    if not any(c.get("uuid") == DGGS_CONVENTION["uuid"] for c in conventions):
        problems.append("zarr_conventions does not declare the DGGS convention v1")
    dggs = attrs.get("dggs")
    if not isinstance(dggs, Mapping):
        return problems + ["no dggs group attribute"]
    for key in ("name", "refinement_level", "spatial_dimension", "indexing_scheme"):
        if key not in dggs:
            problems.append(f"dggs.{key} missing")
    if problems:
        return problems
    level = dggs["refinement_level"]
    scheme = dggs["indexing_scheme"]
    dim = dggs["spatial_dimension"]
    coord = dggs.get("coordinate")
    if dggs["name"] != "healpix":
        problems.append(f"dggs.name is {dggs['name']!r}, expected 'healpix'")

    ell = dggs.get("ellipsoid")
    if ell is not None:
        extra = set(ell) - {
            "name",
            "radius",
            "semi_major_axis",
            "semi_minor_axis",
            "inverse_flattening",
        }
        if extra:
            problems.append(f"dggs.ellipsoid has non-standard keys {sorted(extra)}")

    if coord is not None:
        if dggs.get("compression") is None:
            problems.append("dggs.compression is required when coordinate is set")
        if coord not in variables:
            problems.append(f"dggs.coordinate {coord!r} is not an array of the group")
        else:
            cv = variables[coord]
            if list(cv["dims"]) != [dim]:
                problems.append(
                    f"coordinate {coord!r} has dims {list(cv['dims'])}, "
                    f"expected [{dim!r}]"
                )
            stale = sorted(set(cv["attrs"]) & set(LEGACY_CELL_KEYS))
            if stale:
                problems.append(f"{coord!r} still has old grid attributes {stale}")
            if cv["attrs"].get("standard_name") != "healpix_index":
                problems.append(f"{coord!r} lacks standard_name='healpix_index'")
    if not any(dim in v["dims"] for v in variables.values()):
        problems.append(f"no array uses the spatial dimension {dim!r}")

    gm_names = set()
    for name, var in variables.items():
        if dim in var["dims"] and name != coord:
            gm = var["attrs"].get("grid_mapping")
            if gm is None:
                problems.append(f"{name!r} has no grid_mapping")
                continue
            gm_names.add(gm.split(":")[0])
            if coord is not None and coord != dim:
                if coord not in var["attrs"].get("coordinates", "").split():
                    problems.append(f"{name!r} does not list {coord!r} in coordinates")
    for gm in sorted(gm_names):
        if gm not in variables:
            problems.append(f"grid mapping variable {gm!r} missing")
            continue
        ga = variables[gm]["attrs"]
        expected = {
            "grid_mapping_name": "healpix",
            "indexing_scheme": scheme,
            "refinement_level": level,
        }
        for key, value in expected.items():
            if ga.get(key) != value:
                problems.append(f"{gm}.{key} is {ga.get(key)!r}, expected {value!r}")
        if ell is not None and "semi_major_axis" in ell:
            for key in ("semi_major_axis", "inverse_flattening", "semi_minor_axis"):
                if key in ell and ga.get(key) != ell[key]:
                    problems.append(f"{gm}.{key} differs from dggs.ellipsoid.{key}")
        names = (
            "reference_ellipsoid_name",
            "prime_meridian_name",
            "horizontal_datum_name",
            "geographic_crs_name",
        )
        present = [n for n in names if n in ga]
        if present and len(present) != len(names):
            problems.append(f"{gm}: CF requires all of {names} if any is set")
        if "healpix_nside" in ga and ga["healpix_nside"] != 2**level:
            problems.append(f"{gm}.healpix_nside does not match the refinement level")
    return problems
