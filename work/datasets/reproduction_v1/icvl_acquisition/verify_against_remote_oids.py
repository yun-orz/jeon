#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_against_remote_oids.py -- strongest available integrity check.

While UNauthenticated, huggingface.co redacts the LFS oids of this gated repo on
the wire (every oid is literally 64 asterisks), so the file list and sizes can be
trusted but not the digests. Once authenticated the same endpoint returns the real
64-hex oids.

This script therefore:
  1. re-fetches the pinned revision's mat listing WITH the current credentials;
  2. refuses to pretend it verified anything if the oids are still redacted;
  3. compares every locally computed SHA256 from icvl_file_manifest.json against
     the repository's own published oid for that path.

A match proves the local bytes are bit-identical to what the repository declares,
independently of the download client. A mismatch is a hard failure.

Writes remote_oid_verification.json. Modifies nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import httpx

try:
    from huggingface_hub import get_token
except ImportError:  # pragma: no cover
    get_token = None

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def auth_headers():
    """Bearer header from the locally stored credential.

    The token value is read here and placed straight into a request header. It is
    NEVER printed, logged, or written to any report. If no credential is stored,
    an empty header dict is returned and the caller treats redacted oids as
    'not verified' rather than as a pass.
    """
    if get_token is None:
        return {}, False
    tok = get_token()
    if not tok:
        return {}, False
    return {"Authorization": "Bearer " + tok}, True


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--repo-id", default="ICVL-BGU/ICVL_HS_2016")
    args = ap.parse_args()

    out = args.out_dir

    headers, have_token = auth_headers()

    # ---- 1. re-fetch the tree WITH the stored credential (cheap gate, done first) ----
    # This must be authenticated: on a gated repo huggingface.co redacts every
    # lfs.oid to asterisks for anonymous callers, which would make the comparison
    # below silently vacuous.
    base = (
        "https://huggingface.co/api/datasets/%s/tree/%s/mat?recursive=1&expand=true"
        % (args.repo_id, args.revision)
    )
    entries = []
    url = base
    while url:
        r = httpx.get(url, headers=headers, timeout=90)
        r.raise_for_status()
        entries.extend(x for x in r.json() if x.get("type") == "file")
        link = r.headers.get("link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None

    remote = {}
    redacted = 0
    for e in entries:
        oid = (e.get("lfs") or {}).get("oid", "")
        if set(oid) == {"*"}:
            redacted += 1
        remote[e["path"]] = oid

    report = {
        "artifact": "remote_oid_verification.json",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "credential_used": have_token,
        "credential_value_recorded": False,
        "remote_entry_count": len(entries),
        "oids_redacted": redacted,
    }

    def write_report():
        p = os.path.join(out, "remote_oid_verification.json")
        os.makedirs(out, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        return p

    if redacted or not entries:
        report["status"] = "not_verified_oids_redacted"
        report["explanation"] = (
            "The API still returns redacted (asterisk) oids, which means the request "
            "is not authenticated for this gated repo. No digest comparison was made "
            "and no verification is claimed."
        )
        write_report()
        print("status: not_verified_oids_redacted (oids still redacted: %d)" % redacted)
        return 3

    # ---- 2. need a manifest to compare against ----
    manifest_path = os.path.join(out, "icvl_file_manifest.json")
    if not os.path.exists(manifest_path):
        report["status"] = "not_verified_no_manifest"
        report["explanation"] = (
            "Oids are available but icvl_file_manifest.json does not exist yet, so "
            "there are no locally computed digests to compare. Run the audit first."
        )
        write_report()
        print("status: not_verified_no_manifest (run the audit first)")
        return 4

    manifest = load_json(manifest_path)

    # ---- 3. real oids available: compare every file ----
    matched, mismatched, missing, nothex = [], [], [], []
    for f in manifest["files"]:
        rel = f["relative_path"]
        local_sha = f.get("sha256")
        remote_oid = remote.get(rel)
        if remote_oid is None:
            missing.append(rel)
        elif not HEX64.match(remote_oid):
            nothex.append(rel)
        elif local_sha and local_sha == remote_oid:
            matched.append(rel)
        else:
            mismatched.append(rel)

    report.update(
        {
            "compared_count": len(manifest["files"]),
            "matched_count": len(matched),
            "mismatched_count": len(mismatched),
            "remote_only_missing_count": len(missing),
            "malformed_oid_count": len(nothex),
            "mismatched": mismatched[:50],
            "remote_missing": missing[:50],
            "malformed": nothex[:50],
            "status": "passed" if (not mismatched and not missing and not nothex and matched) else "validation_failed",
        }
    )

    with open(os.path.join(out, "remote_oid_verification.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("status                : %s" % report["status"])
    print("compared              : %d" % report["compared_count"])
    print("matched remote oid    : %d" % report["matched_count"])
    print("mismatched            : %d" % report["mismatched_count"])
    print("remote-side missing   : %d" % report["remote_only_missing_count"])
    print("WROTE remote_oid_verification.json")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
