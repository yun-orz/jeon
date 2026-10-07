#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_icvl_download.py -- ICVL acquisition work package, task doc sections 14-18.

This is the only audit script allowed by the work package ("audit_icvl_download.py").
It may ONLY: read Hugging Face metadata, read the .mat files, compute hashes and
produce reports. It must never modify a raw .mat.

Two reports are produced:

  1. icvl_file_manifest.json  (sections 14 + 15) -- file level
       * the authoritative file list and sizes come from the PINNED revision remote
         tree, not from the local disk, so a truncated file cannot look complete;
       * every expected file must exist, be readable, have size > 0 and match the
         remote size exactly;
       * SHA256 per file, preferring the SHA256 that the `hf` client already
         recorded in <data_root>/.cache/huggingface/download/*.metadata and
         falling back to hashing the file from disk;
       * .part/.incomplete/.tmp leftovers are reported and never counted complete;
       * scene stems must be unique.

  2. icvl_hdf5_audit.json  (sections 16, 17, 18) -- HDF5 metadata level
       * file opens as HDF5; top-level 'rad' and 'bands' exist; 'bands' readable;
       * rad and bands shape AND dtype recorded exactly AS STORED -- the array is
         never transposed before reporting (section 17);
       * the real wavelength values are recorded, checked strictly increasing and
         checked against 400-700 nm / 31 bands / 10 nm;
       * presence of 420-660 nm is verified and the target indices are produced by
         MATCHING REAL WAVELENGTH VALUES, never by a hardcoded slice 2:27 / 3:28
         (section 18);
       * no radiometric normalization of any kind is applied (section 16.15).

The full cube is never loaded: only the small 'bands' array is read. Reading it
also serves as the section 15 proof that the bytes are genuinely available rather
than a cloud placeholder.

Usage:
  python audit_icvl_download.py all          --remote-tree <t> --data-root <d> --out-dir <o> --revision <sha>
  python audit_icvl_download.py manifest     --remote-tree <t> --data-root <d> --out-dir <o> --revision <sha>
  python audit_icvl_download.py hdf5         --manifest <icvl_file_manifest.json> --data-root <d> --out-dir <o> --revision <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

PART_SUFFIXES = (".part", ".incomplete", ".tmp", ".crdownload", ".partial")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def remote_expected_digest(entry: dict):
    """The repository's own declared digest for a tree entry, or None.

    huggingface.co REDACTES this field to 64 asterisks when a gated repo is
    queried without a token, so a value is only returned when it is a real
    64-hex digest. Returning None means 'not available', never 'matches'.
    """
    oid = (entry.get("lfs") or {}).get("oid", "")
    return oid if SHA256_RE.match(oid) else None

TARGET_START_NM = 420
TARGET_STOP_NM = 660
TARGET_STEP_NM = 10
WAVELENGTH_TOL_NM = 1e-6


def expected_targets():
    return list(range(TARGET_START_NM, TARGET_STOP_NM + 1, TARGET_STEP_NM))


def load_json(path: str):
    # utf-8-sig tolerates the BOM added by PowerShell 5.1 'Set-Content -Encoding UTF8'
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def sha256_file(path: str, buf_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:  # read-only
        while True:
            chunk = fh.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_remote_tree(path: str):
    tree = load_json(path)
    files = []
    for e in tree:
        if e.get("type") != "file":
            continue
        p = e.get("path", "")
        if p.startswith("mat/") and p.lower().endswith(".mat"):
            files.append(
                {
                    "path": p,
                    "size": e.get("size"),
                    "expected_sha256": remote_expected_digest(e),
                }
            )
    files.sort(key=lambda x: x["path"])
    return files


def metadata_index(data_root: str) -> dict:
    """remote relative path -> sha256 recorded by the hf client, when available."""
    idx = {}
    meta_dir = os.path.join(data_root, ".cache", "huggingface", "download")
    if not os.path.isdir(meta_dir):
        return idx
    for name in os.listdir(meta_dir):
        if not name.endswith(".metadata"):
            continue
        try:
            with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as fh:
                blob = fh.read()
        except Exception:
            continue
        digest = None
        for tok in blob.replace("\n", " ").split():
            if SHA256_RE.match(tok):
                digest = tok
                break
        if digest:
            idx[name[: -len(".metadata")]] = digest
    return idx


# --------------------------------------------------------------------------- #
# 1. file-level manifest -- sections 14 + 15
# --------------------------------------------------------------------------- #
def build_manifest(args) -> int:
    t0 = time.time()
    remote = load_remote_tree(args.remote_tree)
    meta_idx = metadata_index(args.data_root)

    records, problems, leftovers = [], [], []
    stems = {}
    hash_disk = 0
    digest_checked = digest_ok = digest_bad = 0
    total_bytes = 0

    for dirpath, _dirnames, filenames in os.walk(args.data_root):
        for fn in filenames:
            if fn.lower().endswith(PART_SUFFIXES):
                leftovers.append(os.path.relpath(os.path.join(dirpath, fn), args.data_root))

    for item in remote:
        rel = item["path"]
        local = os.path.join(args.data_root, rel.replace("/", os.sep))
        rec = {
            "relative_path": rel,
            "size_bytes": None,
            "sha256": None,
            "scene_stem": os.path.splitext(os.path.basename(rel))[0],
            "expected_size_bytes": item.get("size"),
            "expected_sha256": item.get("expected_sha256"),
            "sha256_matches_remote": None,
            "size_matches_remote": False,
            "sha256_source": None,
            "readable": False,
        }

        if os.path.isfile(local):
            st = os.stat(local)
            rec["size_bytes"] = st.st_size
            rec["readable"] = os.access(local, os.R_OK)
            rec["size_matches_remote"] = (item.get("size") is None) or (
                st.st_size == item["size"]
            )
            if rec["readable"]:
                # ALWAYS hash the bytes on disk. The value the hf client caches in
                # .cache/huggingface/download/*.metadata is an ETag, which is not
                # guaranteed to be a SHA256 (the Xet backend stores a different
                # digest there), so it is used only as a cross-check below and
                # never as the manifest's sha256.
                rec["sha256"] = sha256_file(local)
                rec["sha256_source"] = "computed_on_disk"
                hash_disk += 1

                cached = meta_idx.get(rel) or meta_idx.get(rel.replace("/", os.sep))
                rec["hf_client_cached_digest"] = cached
                rec["hf_client_digest_is_sha256"] = bool(cached and SHA256_RE.match(cached))
                rec["hf_client_digest_agrees"] = bool(
                    cached and cached == rec["sha256"]
                )

                expected = item.get("expected_sha256")
                if expected:
                    digest_checked += 1
                    rec["sha256_matches_remote"] = (rec["sha256"] == expected)
                    if rec["sha256_matches_remote"]:
                        digest_ok += 1
                    else:
                        digest_bad += 1
                        problems.append(
                            "SHA256_MISMATCH: %s local=%s remote=%s"
                            % (rel, rec["sha256"], expected)
                        )
            total_bytes += st.st_size
        else:
            problems.append("MISSING: %s" % rel)

        if rec["size_bytes"] is not None and not rec["readable"]:
            problems.append("UNREADABLE: %s" % rel)
        if rec["size_bytes"] == 0:
            problems.append("ZERO_SIZE: %s" % rel)
        if rec["size_bytes"] is not None and not rec["size_matches_remote"]:
            problems.append(
                "SIZE_MISMATCH: %s local=%s remote=%s"
                % (rel, rec["size_bytes"], item.get("size"))
            )

        stems.setdefault(rec["scene_stem"], []).append(rel)
        records.append(rec)

    for k, v in stems.items():
        if len(v) > 1:
            problems.append("DUPLICATE_SCENE_STEM: %s -> %s" % (k, v))

    present = sum(1 for r in records if r["size_bytes"] is not None)
    unique_remote = len({r["path"] for r in remote})
    # An empty expected inventory must never read as a pass.
    if not remote:
        problems.append("EMPTY_REMOTE_TREE: no mat/* entries found in the pinned tree")
    status = "validation_failed" if problems else "passed"

    manifest = {
        "artifact": "icvl_file_manifest.json",
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "scope": "mat/* only",
        "data_root": args.data_root,
        "mat_file_count_remote_pinned": unique_remote,
        "mat_file_count_present_local": present,
        "expected_scene_count_doc": args.expected_scenes,
        "count_matches_doc_expectation": (unique_remote == args.expected_scenes),
        "unique_scene_count": len(stems),
        "all_sizes_positive": all((r["size_bytes"] or 0) > 0 for r in records),
        "all_sizes_match_remote": all(r["size_matches_remote"] for r in records),
        "all_sha256_present": all(r["sha256"] for r in records),
        "total_bytes_present": total_bytes,
        "sha256_computed_on_disk": hash_disk,
        "sha256_algorithm": "SHA256, always computed from the bytes on disk",
        "remote_digest_comparison": {
            "comparisons_performed": digest_checked,
            "matched": digest_ok,
            "mismatched": digest_bad,
            "available": digest_checked > 0,
            "note": (
                "The repository's own LFS oid is redacted to asterisks by "
                "huggingface.co while this gated repo is queried without a token, "
                "so no comparison is possible anonymously. verify_against_remote_oids.py "
                "performs this comparison after authentication."
            ) if digest_checked == 0 else "compared against the pinned tree's own digests",
        },
        "partial_or_temp_files": sorted(leftovers),
        "raw_downloaded": False,
        "preview_downloaded": False,
        "problems": problems,
        "elapsed_seconds": round(time.time() - t0, 2),
        "files": records,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "icvl_file_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print("WROTE %s" % out)
    print("status                  : %s" % status)
    print("remote mat/* pinned rev  : %d" % unique_remote)
    print("local mat/* present      : %d" % present)
    print("unique scene stems       : %d" % len(stems))
    print("total bytes present      : %d" % total_bytes)
    print("sha256 computed on disk  : %d" % hash_disk)
    if digest_checked:
        print("remote digest compared   : %d (matched %d, mismatched %d)"
              % (digest_checked, digest_ok, digest_bad))
    else:
        print("remote digest compared   : 0 (repo oids redacted while unauthenticated)")
    print("partial/temp files       : %d" % len(leftovers))
    print("problems                 : %d" % len(problems))
    for p in problems[:40]:
        print("  - %s" % p)
    return 0 if status == "passed" else 2


# --------------------------------------------------------------------------- #
# 2. HDF5 metadata audit -- sections 16, 17, 18
# --------------------------------------------------------------------------- #
def inspect_mat(path: str, rel: str) -> dict:
    rec = {
        "relative_path": rel,
        "scene_stem": os.path.splitext(os.path.basename(rel))[0],
        "hdf5_openable": False,
        "has_rad": False,
        "has_bands": False,
        "bands_readable": False,
        "top_level_keys": None,
        "rad_shape_stored": None,
        "rad_dtype": None,
        "rad_ndim": None,
        "bands_shape_stored": None,
        "bands_dtype": None,
        "bands_nm": None,
        "bands_len": None,
        "wavelength_strictly_increasing": None,
        "is_400_700_31_10nm": None,
        "contains_420_660_10nm": None,
        "target_band_count": None,
        "target_band_indices_420_660": None,
        "rad_band_axis_equals_bands_len": None,
        "cloud_placeholder_or_read_error": False,
        "error": None,
    }
    if h5py is None:
        rec["error"] = "h5py is not available in this interpreter"
        rec["cloud_placeholder_or_read_error"] = True
        return rec

    try:
        with h5py.File(path, "r") as f:  # read-only
            rec["hdf5_openable"] = True
            keys = list(f.keys())
            rec["top_level_keys"] = keys
            rec["has_rad"] = "rad" in f
            rec["has_bands"] = "bands" in f

            # ---- rad: metadata only; payload never touched, never transposed ----
            if rec["has_rad"]:
                ds = f["rad"]
                if isinstance(ds, h5py.Dataset):
                    rec["rad_shape_stored"] = [int(x) for x in ds.shape]
                    rec["rad_ndim"] = int(ds.ndim)
                    rec["rad_dtype"] = str(ds.dtype)
                else:
                    rec["rad_dtype"] = "<non-dataset: %s>" % type(ds).__name__

            # ---- bands: small, read fully ----
            if rec["has_bands"]:
                ds = f["bands"]
                if isinstance(ds, h5py.Dataset):
                    rec["bands_shape_stored"] = [int(x) for x in ds.shape]
                    rec["bands_dtype"] = str(ds.dtype)
                    flat = np.ravel(np.asarray(ds[()]))
                    rec["bands_readable"] = True
                    rec["bands_len"] = int(flat.size)
                    vals = [float(v) for v in flat]
                    rec["bands_nm"] = vals

                    rec["wavelength_strictly_increasing"] = bool(
                        all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
                    )

                    # Tolerance is widened for the structural grid test so that
                    # wavelengths stored as float32 (which cannot represent
                    # 610.0 exactly in a 400..700 sweep) still verify. 1e-3 nm is
                    # three orders of magnitude tighter than the 10 nm spacing and
                    # cannot mask a genuinely wrong grid.
                    grid_tol = max(WAVELENGTH_TOL_NM, 1e-3)
                    exp31 = [400.0 + 10.0 * i for i in range(31)]
                    rec["is_400_700_31_10nm"] = bool(
                        len(vals) == 31
                        and all(
                            abs(vals[i] - exp31[i]) <= grid_tol
                            for i in range(31)
                        )
                    )

                    # ---- section 18: match REAL values, no hardcoded slice ----
                    idx, missing = [], []
                    for t in expected_targets():
                        hit = None
                        for i, v in enumerate(vals):
                            if abs(v - float(t)) <= WAVELENGTH_TOL_NM:
                                hit = i
                                break
                        if hit is None:
                            missing.append(t)
                        else:
                            idx.append(hit)
                    rec["target_band_indices_420_660"] = idx if not missing else None
                    rec["target_band_count"] = len(idx)
                    # Exactly the 25 target bands, in order. Presence of the values
                    # is not enough on its own -- the task doc requires the target
                    # 25 bands, so a file whose grid is off-spec is caught by
                    # is_400_700_31_10nm and re-checked here on count and order.
                    rec["contains_420_660_10nm"] = bool(
                        not missing
                        and len(idx) == len(expected_targets())
                        and all(idx[i] < idx[i + 1] for i in range(len(idx) - 1))
                    )
                    if missing:
                        rec["error"] = "missing target wavelengths (nm): %s" % missing
                    if rec["target_band_count"] != len(expected_targets()):
                        extra = "target band count %d != %d" % (
                            rec["target_band_count"],
                            len(expected_targets()),
                        )
                        rec["error"] = (
                            (rec["error"] + "; ") if rec["error"] else ""
                        ) + extra

                    if rec["rad_shape_stored"]:
                        rec["rad_band_axis_equals_bands_len"] = bool(
                            rec["bands_len"] in rec["rad_shape_stored"]
                        )
                else:
                    rec["bands_dtype"] = "<non-dataset: %s>" % type(ds).__name__
    except Exception as exc:  # OSError here == cloud placeholder / unavailable
        rec["cloud_placeholder_or_read_error"] = True
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["traceback"] = traceback.format_exc(limit=3)

    return rec


def build_hdf5_audit(args) -> int:
    t0 = time.time()
    manifest = load_json(args.manifest)

    scenes, failures = [], []
    n_open = n_rad = n_bands = n_read = n_target = n_placeholder = 0

    for entry in manifest["files"]:
        rel = entry["relative_path"]
        local = os.path.join(args.data_root, rel.replace("/", os.sep))
        rec = inspect_mat(local, rel)
        rec["sha256"] = entry.get("sha256")
        rec["size_bytes"] = entry.get("size_bytes")
        scenes.append(rec)

        n_open += bool(rec["hdf5_openable"])
        n_rad += bool(rec["has_rad"])
        n_bands += bool(rec["has_bands"])
        n_read += bool(rec["bands_readable"])
        n_target += bool(rec["contains_420_660_10nm"])
        n_placeholder += bool(rec["cloud_placeholder_or_read_error"])

        bad = []
        if not rec["hdf5_openable"]:
            bad.append("not HDF5-openable")
        if not rec["has_rad"]:
            bad.append("missing rad")
        if not rec["has_bands"]:
            bad.append("missing bands")
        if not rec["bands_readable"]:
            bad.append("bands unreadable")
        if rec["wavelength_strictly_increasing"] is False:
            bad.append("wavelengths not strictly increasing")
        # The documented grid is 400-700 nm / 31 bands / 10 nm. A file that is
        # merely a DENSE SUBSET of the grid would otherwise sneak through the
        # 25-band test while violating the documented structure, so enforce it.
        if rec["is_400_700_31_10nm"] is False:
            bad.append("not 400-700nm / 31 bands / 10nm")
        if rec["contains_420_660_10nm"] is False:
            bad.append("420-660nm / 25-band check failed")
        if bad:
            failures.append(
                {"relative_path": rel, "issues": bad, "error": rec.get("error")}
            )

    grids = {}
    for s in scenes:
        if s["bands_nm"]:
            grids.setdefault(tuple(s["bands_nm"]), []).append(s["scene_stem"])
    grid_list = [
        {"bands_nm": list(k), "scene_count": len(v), "example_scene": v[0]}
        for k, v in sorted(grids.items())
    ]

    total = len(scenes)
    status = "validation_failed" if (total == 0 or failures) else "passed"
    n_grid = sum(1 for s in scenes if s["is_400_700_31_10nm"])
    n_inc = sum(1 for s in scenes if s["wavelength_strictly_increasing"])

    audit = {
        "artifact": "icvl_hdf5_audit.json",
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "scope": "mat/* only",
        "data_root": args.data_root,
        "mat_file_count": total,
        "unique_scene_count": len({s["scene_stem"] for s in scenes}),
        "all_hdf5_openable": (total > 0 and n_open == total),
        "all_have_rad": (total > 0 and n_rad == total),
        "all_have_bands": (total > 0 and n_bands == total),
        "all_bands_readable": (total > 0 and n_read == total),
        "all_wavelengths_strictly_increasing": (total > 0 and n_inc == total),
        "all_are_400_700_31_10nm": (total > 0 and n_grid == total),
        "all_contain_420_660_10nm": (total > 0 and n_target == total),
        "target_band_indices_source": "value-match against each file's own bands array",
        "rad_shape_stored_not_transposed": True,
        "radiometric_normalization_applied": False,
        "raw_downloaded": False,
        "preview_downloaded": False,
        "cloud_placeholder_detected": n_placeholder > 0,
        "cloud_placeholder_or_read_error_count": n_placeholder,
        "distinct_wavelength_grids": grid_list,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": round(time.time() - t0, 2),
        "scenes": scenes,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "icvl_hdf5_audit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=2)

    print("WROTE %s" % out)
    print("status                        : %s" % status)
    print("mat files audited             : %d" % total)
    print("unique scene stems            : %d" % audit["unique_scene_count"])
    print("hdf5 openable                 : %d/%d" % (n_open, total))
    print("has rad                       : %d/%d" % (n_rad, total))
    print("has bands                     : %d/%d" % (n_bands, total))
    print("bands readable                : %d/%d" % (n_read, total))
    print("wavelengths strictly increase : %d/%d" % (n_inc, total))
    print("is 400-700nm/31band/10nm      : %d/%d" % (n_grid, total))
    print("contains 420-660nm (25 bands) : %d/%d" % (n_target, total))
    print("cloud placeholder / read errs : %d" % n_placeholder)
    print("distinct wavelength grids     : %d" % len(grid_list))
    for g in grid_list:
        print("  - %d scenes, example=%s" % (g["scene_count"], g["example_scene"]))
        print("      bands_nm = %s" % g["bands_nm"])
    if scenes:
        ex = scenes[0]
        print("example scene                 : %s" % ex["scene_stem"])
        print("  rad_shape_stored            : %s" % ex["rad_shape_stored"])
        print("  rad_dtype                   : %s" % ex["rad_dtype"])
        print("  bands_shape_stored          : %s" % ex["bands_shape_stored"])
        print("  target_band_indices_420_660 : %s" % ex["target_band_indices_420_660"])
    print("failures                      : %d" % len(failures))
    for f in failures[:30]:
        print("  - %s: %s | %s" % (f["relative_path"], ", ".join(f["issues"]), f["error"]))
    return 0 if status == "passed" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "manifest", "hdf5"], nargs="?", default="all")
    ap.add_argument("--remote-tree", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--repo-id", default="ICVL-BGU/ICVL_HS_2016")
    ap.add_argument("--expected-scenes", type=int, default=202)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(__file__))
    if args.remote_tree is None:
        args.remote_tree = os.path.join(args.out_dir, "remote_tree_pinned.json")
    if args.manifest is None:
        args.manifest = os.path.join(args.out_dir, "icvl_file_manifest.json")

    rc = 0
    if args.mode in ("all", "manifest"):
        rc |= build_manifest(args)
        print("")
    if args.mode in ("all", "hdf5"):
        rc |= build_hdf5_audit(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
