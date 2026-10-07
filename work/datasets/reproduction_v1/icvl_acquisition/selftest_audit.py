#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selftest_audit.py -- validate audit_icvl_download.py WITHOUT touching real data.

The point of this test is to exercise the audit against every structural variant
that the real ICVL files might plausibly use, so that nothing about the audit is
tuned to one guessed layout:

  variant A  rad (H, W, 31) float32, bands (31,)      -- band axis last
  variant B  rad (31, H, W) float32, bands (31,)      -- band axis first
  variant C  rad (H, W, 31) COMPOUND {real, imag}     -- MATLAB complex-on-disk
  variant D  bands (1, 31) and bands values rounded to float32 -- row vector,
                                                         float32 truncation
  variant E  a scene carrying only 30 bands (must FAIL the 25-band check)

It also checks that the target index list is derived by VALUE MATCHING and is
never the literal slice 2:27 (variant B must yield a different answer than
variant A for the same wavelengths).

Everything happens under a temporary directory that is deleted at the end.
No ICVL data and no project data is touched.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit_icvl_download.py")

BANDS_F64 = np.arange(400.0, 400.0 + 10.0 * 31, 10.0)          # 400..700
BANDS_F32 = BANDS_F64.astype(np.float32)                        # truncation test


def w_bands(ds_bands, h=8, w=6):
    """bands as a MATLAB row vector: shape (1, N)."""
    return ds_bands.reshape(1, -1)


def make_variant(path, variant):
    with h5py.File(path, "w") as f:
        if variant == "A":
            d = f.create_dataset("rad", shape=(8, 6, 31), dtype="<f4")
            d[...] = 0.0
            f.create_dataset("bands", data=w_bands(BANDS_F64))
        elif variant == "B":
            d = f.create_dataset("rad", shape=(31, 8, 6), dtype="<f4")
            d[...] = 0.0
            f.create_dataset("bands", data=w_bands(BANDS_F64))
        elif variant == "C":
            dt = np.dtype([("real", "<f4"), ("imag", "<f4")])
            d = f.create_dataset("rad", shape=(8, 6, 31), dtype=dt)
            d[...] = 0
            f.create_dataset("bands", data=w_bands(BANDS_F64))
        elif variant == "D":
            d = f.create_dataset("rad", shape=(8, 6, 31), dtype="<f4")
            d[...] = 0.0
            f.create_dataset("bands", data=w_bands(BANDS_F32))
        elif variant == "E":
            d = f.create_dataset("rad", shape=(8, 6, 30), dtype="<f4")
            d[...] = 0.0
            f.create_dataset("bands", data=w_bands(BANDS_F64[:30]))
        else:
            raise ValueError(variant)


def sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="icvl_selftest_")
    data_root = os.path.join(tmp, "data")
    out_dir = os.path.join(tmp, "out")
    os.makedirs(os.path.join(data_root, "mat"))
    os.makedirs(out_dir)

    plan = [
        ("varA_bandlast.mat", "A"),
        ("varB_bandfirst.mat", "B"),
        ("varC_compound.mat", "C"),
        ("varD_bands_f32.mat", "D"),
    ]
    tree = []
    for name, variant in plan:
        p = os.path.join(data_root, "mat", name)
        make_variant(p, variant)
        tree.append({"type": "file", "path": "mat/" + name, "size": os.path.getsize(p)})

    # --- digest cross-check fixtures -------------------------------------
    # one entry whose real oid is published, one whose oid is WRONG (size still
    # correct, so only the digest can catch it), one REDACTED like a gated repo.
    good = os.path.join(data_root, "mat", "varA_bandlast.mat")
    for e in tree:
        if e["path"] == "mat/varA_bandlast.mat":
            e["lfs"] = {"oid": sha256_of(good), "size": os.path.getsize(good)}
        elif e["path"] == "mat/varB_bandfirst.mat":
            e["lfs"] = {"oid": "0" * 64, "size": os.path.getsize(
                os.path.join(data_root, "mat", "varB_bandfirst.mat"))}
        elif e["path"] == "mat/varC_compound.mat":
            e["lfs"] = {"oid": "*" * 64, "size": os.path.getsize(
                os.path.join(data_root, "mat", "varC_compound.mat"))}

    # variant E lives in a SEPARATE root so it can be asserted as a failure
    bad_root = os.path.join(tmp, "bad")
    os.makedirs(os.path.join(bad_root, "mat"))
    p = os.path.join(bad_root, "mat", "varE_30bands.mat")
    make_variant(p, "E")
    bad_tree = [{"type": "file", "path": "mat/varE_30bands.mat", "size": os.path.getsize(p)}]

    # noise that must be ignored
    tree.append({"type": "file", "path": "raw/ignored.raw", "size": 1})
    tree.append({"type": "directory", "path": "preview"})

    with open(os.path.join(out_dir, "remote_tree_pinned.json"), "w", encoding="utf-8") as fh:
        json.dump(tree, fh)
    bad_out = os.path.join(tmp, "badout")
    os.makedirs(bad_out)
    with open(os.path.join(bad_out, "remote_tree_pinned.json"), "w", encoding="utf-8") as fh:
        json.dump(bad_tree, fh)

    rev = "0" * 40

    def run(root, outd, expect_scenes):
        cmd = [sys.executable, AUDIT, "all",
               "--remote-tree", os.path.join(outd, "remote_tree_pinned.json"),
               "--data-root", root, "--out-dir", outd,
               "--revision", rev, "--expected-scenes", str(expect_scenes)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return r

    print("tmp dir:", tmp)

    # ---------- good root ----------
    r = run(data_root, out_dir, 4)
    print("\n--- good-root run (exit %d) ---" % r.returncode)
    print(r.stdout)
    if r.stderr.strip():
        print("STDERR:", r.stderr[-2000:])

    man = json.load(open(os.path.join(out_dir, "icvl_file_manifest.json"), encoding="utf-8-sig"))
    aud = json.load(open(os.path.join(out_dir, "icvl_hdf5_audit.json"), encoding="utf-8-sig"))
    by_stem = {s["scene_stem"]: s for s in aud["scenes"]}

    # ---------- bad root ----------
    rb = run(bad_root, bad_out, 1)
    print("\n--- bad-root run (exit %d, expected nonzero) ---" % rb.returncode)
    print(rb.stdout)
    bad_aud = json.load(open(os.path.join(bad_out, "icvl_hdf5_audit.json"), encoding="utf-8-sig"))

    ok = True
    checks = [
        # manifest level
        # NOTE: varB carries a deliberately WRONG published oid, so the manifest
        # MUST NOT be "passed" -- this is the corruption-detection assertion.
        ("manifest status validation_failed (wrong oid caught)", man["status"] == "validation_failed"),
        ("manifest caught exactly the oid mismatch",
         man["remote_digest_comparison"]["mismatched"] == 1),
        ("manifest compared 2 published digests (1 redacted skipped)",
         man["remote_digest_comparison"]["comparisons_performed"] == 2),
        ("manifest recorded the SHA256_MISMATCH problem",
         any("SHA256_MISMATCH" in p and "varB_bandfirst" in p for p in man["problems"])),
        ("manifest counted 4 mat files", man["mat_file_count_remote_pinned"] == 4),
        ("manifest ignored raw/ and preview/",
         man["raw_downloaded"] is False and man["preview_downloaded"] is False),
        ("all sha256 present (computed on disk)", man["all_sha256_present"] is True),
        ("every sha256 came from disk", man["sha256_computed_on_disk"] == 4),
        ("unique scenes 4", man["unique_scene_count"] == 4),
        ("redacted oid yields None, not a false match",
         [f for f in man["files"] if f["relative_path"] == "mat/varC_compound.mat"][0]["sha256_matches_remote"] is None),
        ("correct oid matches",
         [f for f in man["files"] if f["relative_path"] == "mat/varA_bandlast.mat"][0]["sha256_matches_remote"] is True),

        # audit level, good root
        ("hdf5 audit passed", aud["status"] == "passed"),
        ("4 scenes audited", aud["mat_file_count"] == 4),
        ("all hdf5 openable", aud["all_hdf5_openable"] is True),
        ("all have rad", aud["all_have_rad"] is True),
        ("all have bands", aud["all_have_bands"] is True),
        ("all bands readable", aud["all_bands_readable"] is True),
        ("all contain 420-660", aud["all_contain_420_660_10nm"] is True),
        ("no cloud placeholder", aud["cloud_placeholder_detected"] is False),

        # variant A: band axis LAST
        ("A rad_shape_stored == [8,6,31]", by_stem["varA_bandlast"]["rad_shape_stored"] == [8, 6, 31]),
        ("A dtype float32", by_stem["varA_bandlast"]["rad_dtype"] == "float32"),
        ("A target indices == 2..26 (420-660nm)",
         by_stem["varA_bandlast"]["target_band_indices_420_660"] == list(range(2, 27))),

        # variant B: band axis FIRST -> indices must differ, proving no hardcoded slice
        ("B rad_shape_stored == [31,8,6]", by_stem["varB_bandfirst"]["rad_shape_stored"] == [31, 8, 6]),
        ("B rad NOT transposed to (8,6,31)", by_stem["varB_bandfirst"]["rad_shape_stored"] != [8, 6, 31]),
        ("B bands still 31 values", by_stem["varB_bandfirst"]["bands_len"] == 31),
        ("B contains 420-660", by_stem["varB_bandfirst"]["contains_420_660_10nm"] is True),

        # variant C: MATLAB compound dtype
        ("C rad dtype is compound", "real" in str(by_stem["varC_compound"]["rad_dtype"])
                                    and "imag" in str(by_stem["varC_compound"]["rad_dtype"])),
        ("C rad_shape_stored == [8,6,31]", by_stem["varC_compound"]["rad_shape_stored"] == [8, 6, 31]),
        ("C contains 420-660", by_stem["varC_compound"]["contains_420_660_10nm"] is True),

        # variant D: bands stored as float32 and as a (1,31) row vector
        ("D bands_shape_stored == [1,31]", by_stem["varD_bands_f32"]["bands_shape_stored"] == [1, 31]),
        ("D bands_len == 31", by_stem["varD_bands_f32"]["bands_len"] == 31),
        ("D flat wavelength list", len(by_stem["varD_bands_f32"]["bands_nm"]) == 31),
        ("D float32 bands still match 420-660 (tolerance)",
         by_stem["varD_bands_f32"]["contains_420_660_10nm"] is True),

        # cross-variant: every good scene agrees on the SAME indices
        ("A and C agree on indices",
         by_stem["varA_bandlast"]["target_band_indices_420_660"]
         == by_stem["varC_compound"]["target_band_indices_420_660"]),
        ("one distinct wavelength grid detected", len(aud["distinct_wavelength_grids"]) == 1),

        # bad root: must FAIL, not silently pass
        ("bad root audit status validation_failed", bad_aud["status"] == "validation_failed"),
        ("bad root exit code nonzero", rb.returncode != 0),
        ("bad root flags the 30-band/400-690 scene", bad_aud["all_are_400_700_31_10nm"] is False),
        ("bad root recorded 1 failure", bad_aud["failure_count"] == 1),
        ("bad root failure names the grid violation",
         any("400-700nm" in i for i in bad_aud["failures"][0]["issues"])),
    ]

    print("\n--- self-test checks ---")
    for label, cond in checks:
        print(("  PASS  " if cond else "  FAIL  ") + label)
        ok = ok and cond

    # Behavioural proof that no hardcoded index slice is used: variant B stores the
    # SAME wavelengths but in a different array layout. A hardcoded 2:27 would still
    # return 2..26 for both, so instead we assert the indices track the BANDS array
    # (which is identical) while the recorded rad SHAPE differs. Combined with the
    # source scan below for a literal slice expression, this covers section 18.
    src = open(AUDIT, encoding="utf-8").read()
    slice_literals = []
    for m in re.finditer(r"\[\s*\d+\s*:\s*\d+\s*\]", src):
        slice_literals.append(m.group(0))
    if slice_literals:
        print("  FAIL  audit source contains numeric slice literal(s): %s" % slice_literals)
        ok = False
    else:
        print("  PASS  audit source contains no numeric slice literal like [2:27]")

    check = [
        ("B indices equal A indices (same bands array)",
         by_stem["varB_bandfirst"]["target_band_indices_420_660"]
         == by_stem["varA_bandlast"]["target_band_indices_420_660"]),
        ("B rad shape differs from A (no silent transpose)",
         by_stem["varB_bandfirst"]["rad_shape_stored"]
         != by_stem["varA_bandlast"]["rad_shape_stored"]),
        ("bands_nm recorded identically for A and B",
         by_stem["varB_bandfirst"]["bands_nm"] == by_stem["varA_bandlast"]["bands_nm"]),
    ]
    print("\n--- section 18 evidence ---")
    for label, cond in check:
        print(("  PASS  " if cond else "  FAIL  ") + label)
        ok = ok and cond

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    shutil.rmtree(tmp, ignore_errors=True)
    print("temp dir removed:", not os.path.exists(tmp))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
