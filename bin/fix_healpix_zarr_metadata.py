#!/usr/bin/env python3
"""Rewrite the HEALPix grid metadata of an existing Zarr v3 store.

Only ``zarr.json`` files are rewritten; chunk files are never opened.  The
group attributes, the cell coordinate, the grid mapping variable and the data
variables get the metadata built by :mod:`healpix_regrid.metadata`, and the
consolidated metadata of the root group is updated to match.

Safety:

* every rewritten ``zarr.json`` is written to a temporary file and moved into
  place with ``os.replace``, so a hard-linked metadata file gets a new inode
  and the other link (for example the original store) is not changed;
* a ``zarr.json`` that is a symbolic link, or that resolves outside the store,
  is refused (chunk directories may be symbolic links, they are not touched);
* the previous metadata files are saved under ``--backup-dir`` first.

Runs with the Python standard library only (Python >= 3.9), e.g. on an HPC
login node::

    python3 bin/fix_healpix_zarr_metadata.py \\
        /path/to/store.zarr --backup-dir /path/to/backup --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil


def _load_metadata_module():
    # import healpix_regrid/metadata.py on its own: the package __init__ pulls
    # in dask, healpix-geo, ... which are not needed (nor installed) here
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "healpix_regrid",
        "metadata.py",
    )
    spec = importlib.util.spec_from_file_location("healpix_regrid_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_md = _load_metadata_module()
cell_coordinate_attrs = _md.cell_coordinate_attrs
check_healpix_metadata = _md.check_healpix_metadata
grid_mapping_attrs = _md.grid_mapping_attrs
group_attrs = _md.group_attrs
LEGACY_CELL_KEYS = _md.LEGACY_CELL_KEYS


def _load(path):
    with open(path) as f:
        return json.load(f)


def _children(root):
    out = {}
    for name in sorted(os.listdir(root)):
        meta = os.path.join(root, name, "zarr.json")
        if os.path.isfile(meta):
            out[name] = _load(meta)
    return out


def _check_writable(store, path):
    if os.path.islink(path):
        raise RuntimeError(f"refusing to write through a symbolic link: {path}")
    real_store = os.path.realpath(store)
    real = os.path.realpath(path)
    if os.path.commonpath([real_store, real]) != real_store:
        raise RuntimeError(f"{path} resolves outside the store ({real})")


def _write_json(store, path, data):
    _check_writable(store, path)
    tmp = path + ".tmp-fix-healpix"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, allow_nan=True)
        f.write("\n")
    os.replace(tmp, path)


def _infer(root_meta, arrays, args):
    attrs = root_meta.get("attributes", {})
    old = attrs.get("dggs") or {}
    crs_candidates = sorted(
        {
            str(m.get("attributes", {}).get("grid_mapping")).split(":")[0]
            for m in arrays.values()
            if m.get("attributes", {}).get("grid_mapping")
        }
    )
    crs_name = args.crs or (crs_candidates[0] if crs_candidates else "crs")
    crs_attrs = arrays.get(crs_name, {}).get("attributes", {})

    dim = args.dim or old.get("spatial_dimension")
    coord = args.coordinate
    if coord is None:
        for cand in (old.get("coordinate"), dim, "cell_ids", "cell", "cells"):
            if cand in arrays and len(arrays[cand].get("dimension_names") or []) == 1:
                coord = cand
                break
    if coord is None:
        raise SystemExit("cannot find the cell coordinate, pass --coordinate")
    cdims = arrays[coord].get("dimension_names") or []
    if dim is None:
        dim = cdims[0]
    if list(cdims) != [dim]:
        raise SystemExit(f"coordinate {coord!r} has dims {cdims}, expected [{dim!r}]")
    if not arrays[coord]["data_type"].startswith(("int", "uint")):
        raise SystemExit(f"coordinate {coord!r} is not an integer array")

    cell_attrs = arrays[coord].get("attributes", {})
    levels = {
        "dggs.refinement_level": old.get("refinement_level"),
        "crs.refinement_level": crs_attrs.get("refinement_level"),
        "cell.level": cell_attrs.get("level"),
        "argument": args.level,
    }
    nside = crs_attrs.get("healpix_nside") or attrs.get("nside")
    if nside:
        levels["crs.healpix_nside"] = int(nside).bit_length() - 1
    found = {k: v for k, v in levels.items() if v is not None}
    if not found:
        raise SystemExit("no refinement level in the store, pass --level")
    if len(set(found.values())) != 1:
        raise SystemExit(f"inconsistent refinement levels: {found}")
    level = int(next(iter(found.values())))

    schemes = {
        "dggs": old.get("indexing_scheme"),
        "crs": crs_attrs.get("indexing_scheme") or crs_attrs.get("healpix_order"),
        "cell": cell_attrs.get("indexing_scheme"),
    }
    if cell_attrs.get("nest") is not None:
        schemes["cell.nest"] = "nested" if cell_attrs["nest"] else "ring"
    found = {k: v for k, v in schemes.items() if v is not None}
    if len(set(found.values())) > 1:
        raise SystemExit(f"inconsistent indexing schemes: {found}")
    scheme = next(iter(found.values()), None) or args.indexing_scheme

    ell = old.get("ellipsoid") or {}
    ell_name = ell.get("name") or cell_attrs.get("ellipsoid") or args.ellipsoid
    semi_major = ell.get("semi_major_axis", ell.get("semimajor_axis"))
    if ell_name.upper() != "WGS84" or (
        semi_major is not None and float(semi_major) != 6378137.0
    ):
        raise SystemExit(f"unexpected ellipsoid {ell}")
    return level, scheme, dim, coord, crs_name


def fix_store(args):
    store = args.store.rstrip("/")
    root_path = os.path.join(store, "zarr.json")
    root_meta = _load(root_path)
    if root_meta.get("zarr_format") != 3 or root_meta.get("node_type") != "group":
        raise SystemExit("only Zarr v3 groups are supported")
    arrays = {
        k: v for k, v in _children(store).items() if v.get("node_type") == "array"
    }
    level, scheme, dim, coord, crs_name = _infer(root_meta, arrays, args)
    print(
        f"level={level} indexing_scheme={scheme} spatial_dimension={dim} "
        f"coordinate={coord} grid_mapping={crs_name}"
    )

    new = {}  # node name ('' for the root) -> new metadata
    # cell coordinate
    m = json.loads(json.dumps(arrays[coord]))
    a = m.setdefault("attributes", {})
    for key in LEGACY_CELL_KEYS:  # superseded by the dggs group attribute
        a.pop(key, None)
    a.update(cell_coordinate_attrs(level, scheme))
    new[coord] = m

    # grid mapping variable
    if crs_name in arrays:
        m = json.loads(json.dumps(arrays[crs_name]))
        if m.get("shape") not in ([], None):
            raise SystemExit(f"grid mapping variable {crs_name!r} is not a scalar")
    else:
        m = {
            "shape": [],
            "data_type": "int8",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {},
            "zarr_format": 3,
            "node_type": "array",
            "storage_transformers": [],
        }
    m["attributes"].update(grid_mapping_attrs(level, scheme, legacy=True))
    new[crs_name] = m

    # data variables
    for name, meta in arrays.items():
        if name in (coord, crs_name) or dim not in (meta.get("dimension_names") or []):
            continue
        m = json.loads(json.dumps(meta))
        a = m.setdefault("attributes", {})
        if a.get("grid_mapping", crs_name).split(":")[0] != crs_name:
            raise SystemExit(f"{name!r} uses another grid mapping: {a['grid_mapping']}")
        a.setdefault("grid_mapping", crs_name)
        if coord != dim:
            coords = a.get("coordinates", "").split()
            if coord not in coords:
                a["coordinates"] = " ".join(coords + [coord])
        if m != meta:
            new[name] = m

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    root_new = json.loads(json.dumps(root_meta))
    root_new["attributes"] = group_attrs(
        root_meta.get("attributes", {}),
        level,
        spatial_dimension=dim,
        coordinate=coord,
        indexing_scheme=scheme,
        history=(
            f"{stamp}: HEALPix grid metadata rewritten by "
            "softwares_RIOMAR/bin/fix_healpix_zarr_metadata.py (array data unchanged)"
        ),
    )
    cm = root_new.get("consolidated_metadata")
    if cm is not None:
        for name, meta in new.items():
            cm["metadata"][name] = meta

    # check before writing
    variables = {
        name: {
            "dims": meta.get("dimension_names") or [],
            "shape": meta.get("shape"),
            "attrs": meta.get("attributes", {}),
        }
        for name, meta in {**arrays, **new}.items()
    }
    problems = check_healpix_metadata(root_new["attributes"], variables)
    if problems:
        raise SystemExit("metadata check failed:\n  " + "\n  ".join(problems))

    changed = [n for n in new if new[n] != arrays.get(n)]

    def _without_history(attrs):
        return {k: v for k, v in attrs.items() if k != "history"}

    if not changed and _without_history(root_new["attributes"]) == _without_history(
        root_meta.get("attributes", {})
    ):
        print("metadata already up to date, nothing to do")
        return
    print("nodes to rewrite:", ["/"] + changed)
    if args.dry_run:
        for n in changed:
            before = arrays.get(n, {}).get("attributes", {})
            after = new[n]["attributes"]
            print(f"--- {n}")
            for k in sorted(set(before) | set(after)):
                if before.get(k) != after.get(k):
                    print(f"   {k}: {before.get(k)!r} -> {after.get(k)!r}")
        before = root_meta.get("attributes", {})
        print("--- / (group attributes)")
        for k in sorted(set(before) | set(root_new["attributes"])):
            if before.get(k) != root_new["attributes"].get(k):
                print(f"   {k}: {before.get(k)!r} -> {root_new['attributes'].get(k)!r}")
        return

    # backup, then write children first and the root (consolidated) last
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=False)
        shutil.copy2(root_path, os.path.join(args.backup_dir, "zarr.json"))
        for n in changed:
            src = os.path.join(store, n, "zarr.json")
            if os.path.exists(src):
                os.makedirs(os.path.join(args.backup_dir, n))
                shutil.copy2(src, os.path.join(args.backup_dir, n, "zarr.json"))
    for n in changed:
        os.makedirs(os.path.join(store, n), exist_ok=True)
        _write_json(store, os.path.join(store, n, "zarr.json"), new[n])
    _write_json(store, root_path, root_new)
    print("done")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("store", help="path of the Zarr v3 store (root group)")
    p.add_argument("--level", type=int, help="refinement level, if not in the store")
    p.add_argument("--dim", help="spatial dimension name")
    p.add_argument("--coordinate", help="cell coordinate name")
    p.add_argument("--crs", help="grid mapping variable name (default: crs)")
    p.add_argument("--indexing-scheme", default="nested", choices=["nested", "ring"])
    p.add_argument("--ellipsoid", default="WGS84")
    p.add_argument("--backup-dir", help="save the previous metadata files here")
    p.add_argument("--dry-run", action="store_true")
    fix_store(p.parse_args(argv))


if __name__ == "__main__":
    main()
