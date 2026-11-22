"""HEALPix grid metadata: Zarr DGGS convention v1 and CF 1.13 healpix."""

import hashlib
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from healpix_regrid.healpix import write_zarr_chunked
from healpix_regrid.metadata import (
    DGGS_CONVENTION,
    apply_healpix_metadata,
    cell_coordinate_attrs,
    check_healpix_metadata,
    dggs_attrs,
    grid_mapping_attrs,
)

ROOT = Path(__file__).resolve().parents[1]
# https://raw.githubusercontent.com/zarr-conventions/dggs/v1/schema.json
SCHEMA = json.loads((ROOT / "tests" / "data" / "dggs_v1_schema.json").read_text())

# writing the Zarr v3 format needs zarr-python >= 3 (Python >= 3.11)
requires_zarr3 = pytest.mark.skipif(
    int(zarr.__version__.split(".")[0]) < 3, reason="Zarr v3 format needs zarr >= 3"
)

PARENT_LEVEL = 1
LEVEL = 3
NCHILD = 4 ** (LEVEL - PARENT_LEVEL)


def _cell_ids(parents=(5, 6, 20)):
    return np.concatenate([np.arange(p * NCHILD, (p + 1) * NCHILD) for p in parents])


def _dataset(nt=6, nz=2):
    cells = _cell_ids()
    rng = np.random.default_rng(0)
    temp = rng.normal(size=(nt, nz, cells.size))
    temp[:, :, ::7] = np.nan
    return xr.Dataset(
        {
            "temp": (
                ("time_counter", "s_rho", "cell_ids"),
                temp,
                {"units": "Celsius", "long_name": "potential temperature"},
            ),
            "zeta": (("time_counter", "cell_ids"), rng.normal(size=(nt, cells.size))),
            "time_only": (("time_counter",), np.arange(nt, dtype="f8")),
        },
        coords={
            "cell_ids": ("cell_ids", cells.astype("int64")),
            "time_counter": np.arange(nt) * 3600.0,
            "s_rho": ("s_rho", np.linspace(-1, 0, nz)),
        },
        attrs={"title": "GAMAR test", "Conventions": "CF-1.6"},
    )


def _schema_errors(group_json):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft7Validator(SCHEMA)
    return [e.message for e in validator.iter_errors(group_json)]


def _variables(ds):
    return {
        name: {"dims": list(var.dims), "shape": list(var.shape), "attrs": var.attrs}
        for name, var in ds.variables.items()
    }


# ---------------------------------------------------------------- builders


def test_dggs_attrs_follow_schema():
    group = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "zarr_conventions": [DGGS_CONVENTION],
            "dggs": dggs_attrs(13, spatial_dimension="cell", coordinate="cell"),
        },
    }
    assert _schema_errors(group) == []
    ell = group["attributes"]["dggs"]["ellipsoid"]
    assert ell == {
        "name": "WGS84",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }


def test_legacy_semimajor_axis_fails_schema():
    dggs = dggs_attrs(13, spatial_dimension="cell", coordinate="cell")
    dggs["ellipsoid"]["semimajor_axis"] = dggs["ellipsoid"].pop("semi_major_axis")
    group = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"zarr_conventions": [DGGS_CONVENTION], "dggs": dggs},
    }
    assert _schema_errors(group)


def test_grid_mapping_attrs():
    attrs = grid_mapping_attrs(13)
    assert attrs["grid_mapping_name"] == "healpix"
    assert attrs["refinement_level"] == 13
    assert attrs["indexing_scheme"] == "nested"
    assert attrs["semi_major_axis"] == 6378137.0
    assert attrs["inverse_flattening"] == 298.257223563
    # read by the RiOMar dashboard
    assert attrs["healpix_nside"] == 2**13
    assert attrs["healpix_order"] == "nested"
    assert "healpix_nside" not in grid_mapping_attrs(13, legacy=False)


@pytest.mark.parametrize(
    "level, scheme", [(-1, "nested"), (30, "nested"), (3, "zuniq"), (True, "nested")]
)
def test_invalid_grid(level, scheme):
    with pytest.raises((ValueError, TypeError)):
        dggs_attrs(
            level, spatial_dimension="cell", coordinate="cell", indexing_scheme=scheme
        )


def test_cell_coordinate_attrs():
    attrs = cell_coordinate_attrs(13)
    assert attrs["standard_name"] == "healpix_index"
    assert set(attrs) == {"standard_name", "long_name"}


# ----------------------------------------------------- apply on a Dataset


def test_apply_healpix_metadata():
    ds = _dataset()
    out = apply_healpix_metadata(ds, level=LEVEL)

    assert check_healpix_metadata(out.attrs, _variables(out)) == []
    assert out.attrs["dggs"]["refinement_level"] == LEVEL
    assert out.attrs["dggs"]["spatial_dimension"] == "cell_ids"
    assert out.attrs["dggs"]["coordinate"] == "cell_ids"
    assert out.attrs["zarr_conventions"] == [DGGS_CONVENTION]
    assert out["crs"].attrs["refinement_level"] == LEVEL
    assert out["cell_ids"].attrs["standard_name"] == "healpix_index"
    assert out["temp"].attrs["grid_mapping"] == "crs"
    assert out["zeta"].attrs["grid_mapping"] == "crs"
    assert "grid_mapping" not in out["time_only"].attrs
    assert out["temp"].attrs["units"] == "Celsius"
    # the source CF-1.6 declaration is dropped and recorded in history
    assert "Conventions" not in out.attrs
    assert "CF-1.6" in out.attrs["history"]
    # the input is not modified, the values are the same objects
    assert "dggs" not in ds.attrs and "standard_name" not in ds["cell_ids"].attrs
    # old per-coordinate attributes are replaced by the dggs group attribute
    legacy = apply_healpix_metadata(
        ds.assign_coords(
            cell_ids=ds["cell_ids"].assign_attrs(grid_name="healpix", level=LEVEL)
        ),
        level=LEVEL,
    )
    assert "grid_name" not in legacy["cell_ids"].attrs
    xr.testing.assert_identical(out["temp"].drop_attrs(), ds["temp"].drop_attrs())


def test_apply_is_idempotent():
    once = apply_healpix_metadata(_dataset(), level=LEVEL)
    twice = apply_healpix_metadata(once, level=LEVEL)
    assert twice.attrs["zarr_conventions"] == [DGGS_CONVENTION]
    assert twice.attrs["dggs"] == once.attrs["dggs"]
    assert twice["crs"].attrs == once["crs"].attrs


def test_cell_coordinate_on_other_dimension():
    ds = _dataset().rename_dims(cell_ids="cell").rename_vars(cell_ids="cell_index")
    out = apply_healpix_metadata(ds, level=LEVEL, cell_coord="cell_index")
    assert out.attrs["dggs"]["spatial_dimension"] == "cell"
    assert out.attrs["dggs"]["coordinate"] == "cell_index"
    assert "cell_index" in out["temp"].attrs["coordinates"].split()
    assert check_healpix_metadata(out.attrs, _variables(out)) == []


def test_reject_bad_coordinate():
    ds = _dataset().assign_coords(cell_ids=_cell_ids().astype("float64"))
    with pytest.raises(TypeError):
        apply_healpix_metadata(ds, level=LEVEL)
    with pytest.raises(KeyError):
        apply_healpix_metadata(_dataset(), level=LEVEL, cell_coord="cells")


def test_check_reports_problems():
    out = apply_healpix_metadata(_dataset(), level=LEVEL)
    out.attrs["dggs"] = dict(out.attrs["dggs"], coordinate="cell")
    out["crs"].attrs["refinement_level"] = LEVEL + 1
    del out["zeta"].attrs["grid_mapping"]
    problems = check_healpix_metadata(out.attrs, _variables(out))
    assert any("'cell' is not an array" in p for p in problems)
    assert any("crs.refinement_level" in p for p in problems)
    assert any("'zeta' has no grid_mapping" in p for p in problems)


# ------------------------------------------------- write / read round trip


def _group_json(store):
    store = Path(store)
    if (store / "zarr.json").exists():
        return json.loads((store / "zarr.json").read_text())
    return {
        "zarr_format": 2,
        "node_type": "group",
        "attributes": json.loads((store / ".zattrs").read_text()),
    }


@pytest.mark.parametrize("zarr_format", [2, pytest.param(3, marks=requires_zarr3)])
def test_zarr_round_trip(tmp_path, zarr_format):
    ds = apply_healpix_metadata(_dataset(), level=LEVEL).chunk(
        {"cell_ids": NCHILD, "time_counter": 2}
    )
    path = tmp_path / "out.zarr"
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=zarr_format)

    back = xr.open_zarr(path, consolidated=True)
    xr.testing.assert_identical(back.compute(), ds.compute())
    assert check_healpix_metadata(back.attrs, _variables(back)) == []
    assert _schema_errors(_group_json(path)) == []

    # consolidated metadata matches the per-node metadata
    plain = xr.open_zarr(path, consolidated=False)
    for name in plain.variables:
        assert plain[name].attrs == back[name].attrs, name
    assert plain.attrs == back.attrs

    # chunks along the spatial dimension are complete parent cells
    assert back["temp"].encoding["chunks"][-1] == NCHILD
    cells = back["cell_ids"].values.reshape(-1, NCHILD)
    assert (np.unique(cells // NCHILD, axis=1).shape[1]) == 1


def test_write_zarr_chunked_keeps_metadata(tmp_path):
    ds = _dataset(nt=7)

    def regrid(block):
        # stands for to_healpix: metadata, then reindex + chunk
        out = apply_healpix_metadata(block, level=LEVEL)
        target = _cell_ids()
        return out.reindex(cell_ids=target, fill_value=np.nan).chunk(
            {"cell_ids": NCHILD, "time_counter": 3}
        )

    path = str(tmp_path / "chunked.zarr")
    write_zarr_chunked(ds, path, regrid_fn=partial(regrid), block_size=3)
    back = xr.open_zarr(path, consolidated=True)
    assert back.sizes["time_counter"] == 7
    np.testing.assert_array_equal(back["temp"].values, ds["temp"].values)
    assert check_healpix_metadata(back.attrs, _variables(back)) == []
    assert back.attrs["dggs"]["refinement_level"] == LEVEL


@requires_zarr3
def test_xdggs_decodes_output(tmp_path):
    xdggs = pytest.importorskip("xdggs")
    ds = apply_healpix_metadata(_dataset(), level=LEVEL)
    path = tmp_path / "out.zarr"
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=3)
    back = xr.open_zarr(path)
    try:
        decoded = xdggs.decode(back, convention="zarr")
    except TypeError:  # xdggs without the convention keyword
        pytest.skip("xdggs version without the Zarr DGGS convention")
    assert decoded.dggs.grid_info.level == LEVEL
    assert decoded.dggs.grid_info.indexing_scheme == "nested"
    # default call, as in the notebooks: the convention is inferred
    assert xdggs.decode(back).dggs.grid_info.level == LEVEL


# ------------------------------------------ fixing an existing Zarr v3 store


FIX_SCRIPT = ROOT / "bin" / "fix_healpix_zarr_metadata.py"


def _legacy_store(path):
    """A store written like the 2026-06 GAMAR_OXY HEALPix test outputs."""
    ds = _dataset().rename({"cell_ids": "cell"})
    ds["crs"] = (
        (),
        np.int64(0),
        {
            "grid_mapping_name": "healpix",
            "healpix_nside": 2**LEVEL,
            "healpix_order": "nested",
        },
    )
    for name in ("temp", "zeta"):
        ds[name].attrs["grid_mapping"] = "crs"
    ds.attrs["dggs"] = {
        "name": "healpix",
        "refinement_level": LEVEL,
        "indexing_scheme": "nested",
        "spatial_dimension": "cell",
        "ellipsoid": {
            "name": "wgs84",
            "semimajor_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
        "coordinate": "cell_ids",
        "compression": "none",
    }
    ds.chunk({"cell": NCHILD}).to_zarr(path, mode="w", consolidated=True, zarr_format=3)
    return ds


def _hashes(path, metadata):
    out = {}
    for root, _, files in os.walk(path):
        for f in files:
            if (f == "zarr.json") == metadata:
                p = os.path.join(root, f)
                digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
                out[os.path.relpath(p, path)] = digest
    return out


def _run_fix(*args):
    return subprocess.run(
        [sys.executable, str(FIX_SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
    )


@requires_zarr3
def test_fix_existing_store(tmp_path):
    store = tmp_path / "legacy.zarr"
    original = _legacy_store(store)
    before_meta = json.loads((store / "zarr.json").read_text())
    assert _schema_errors(before_meta)  # semimajor_axis, no zarr_conventions
    chunks_before = _hashes(store, metadata=False)

    # hard-linked copy of the store: rewriting must not reach the original
    linked = tmp_path / "linked.zarr"
    for root, _, files in os.walk(store):
        dst = linked / Path(root).relative_to(store)
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            os.link(os.path.join(root, f), dst / f)
    meta_before = _hashes(store, metadata=True)

    res = _run_fix(linked, "--dry-run")
    assert res.returncode == 0, res.stderr
    assert _hashes(linked, metadata=True) == meta_before

    res = _run_fix(linked, "--backup-dir", tmp_path / "backup")
    assert res.returncode == 0, res.stderr

    assert _hashes(store, metadata=True) == meta_before  # original untouched
    assert _hashes(linked, metadata=False) == chunks_before  # chunks untouched
    assert (tmp_path / "backup" / "zarr.json").read_bytes() == (
        store / "zarr.json"
    ).read_bytes()

    after = json.loads((linked / "zarr.json").read_text())
    assert _schema_errors(after) == []
    assert after["attributes"]["dggs"]["coordinate"] == "cell"
    assert "Conventions" not in after["attributes"]

    # running it again changes nothing
    meta_fixed = _hashes(linked, metadata=True)
    res = _run_fix(linked)
    assert res.returncode == 0 and "up to date" in res.stdout, res.stderr
    assert _hashes(linked, metadata=True) == meta_fixed

    back = xr.open_zarr(linked, consolidated=True)
    plain = xr.open_zarr(linked, consolidated=False)
    assert check_healpix_metadata(back.attrs, _variables(back)) == []
    for name in plain.variables:
        assert plain[name].attrs == back[name].attrs, name
    assert back["crs"].attrs["healpix_nside"] == 2**LEVEL  # kept for the dashboard
    assert back["cell"].attrs["standard_name"] == "healpix_index"
    for name in ("temp", "zeta", "cell", "time_counter"):
        xr.testing.assert_equal(back[name].drop_attrs(), original[name].drop_attrs())


@requires_zarr3
def test_fix_refuses_symlinked_metadata(tmp_path):
    store = tmp_path / "legacy.zarr"
    _legacy_store(store)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text((store / "cell" / "zarr.json").read_text())
    (store / "cell" / "zarr.json").unlink()
    (store / "cell" / "zarr.json").symlink_to(elsewhere)
    before = elsewhere.read_bytes()

    res = _run_fix(store)
    assert res.returncode != 0
    assert "symbolic link" in res.stderr
    assert elsewhere.read_bytes() == before


@requires_zarr3
def test_fix_detects_inconsistent_levels(tmp_path):
    store = tmp_path / "legacy.zarr"
    _legacy_store(store)
    group = zarr.open_group(store, mode="r+")
    group["crs"].attrs["healpix_nside"] = 2 ** (LEVEL + 1)
    zarr.consolidate_metadata(store)
    res = _run_fix(store)
    assert res.returncode != 0
    assert "inconsistent refinement levels" in res.stderr


@requires_zarr3
def test_fix_store_with_cell_attrs_only(tmp_path):
    """Stores written before this change: grid attributes on cell_ids only."""
    store = tmp_path / "old.zarr"
    ds = _dataset()
    ds["cell_ids"].attrs.update(
        {
            "grid_name": "healpix",
            "level": LEVEL,
            "indexing_scheme": "nested",
            "ellipsoid": "WGS84",
        }
    )
    ds.chunk({"cell_ids": NCHILD}).to_zarr(
        store, mode="w", consolidated=True, zarr_format=3
    )
    chunks_before = _hashes(store, metadata=False)

    res = _run_fix(store)
    assert res.returncode == 0, res.stderr
    assert _hashes(store, metadata=False) == chunks_before
    assert _schema_errors(json.loads((store / "zarr.json").read_text())) == []
    back = xr.open_zarr(store, consolidated=True)
    assert check_healpix_metadata(back.attrs, _variables(back)) == []
    assert back["crs"].attrs["refinement_level"] == LEVEL
    assert back["temp"].attrs["grid_mapping"] == "crs"
    assert "grid_name" not in back["cell_ids"].attrs
