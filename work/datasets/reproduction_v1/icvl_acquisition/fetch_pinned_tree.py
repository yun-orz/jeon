#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_pinned_tree.py -- capture the pinned revision's mat/* tree WITH credentials.

Why this exists: huggingface.co redacts every lfs.oid to 64 asterisks when a
GATED repo is queried without a token. Downloading the tree file anonymously
therefore silently produces a tree whose per-file digests are useless, and the
manifest's remote-digest comparison would then be vacuous.

So the canonical tree capture must be authenticated. The token is read from the
local credential store and placed straight into a request header; it is never
printed, logged, or written into any output file.

Writes:
  remote_tree_pinned.json   full tree (all paths, mat + raw + preview)
  remote_mat_files.json     mat/* files only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import httpx

try:
    from huggingface_hub import get_token
except ImportError:  # pragma: no cover
    get_token = None

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo-id", default="ICVL-BGU/ICVL_HS_2016")
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()

    tok = get_token() if get_token else None
    headers = {"Authorization": "Bearer " + tok} if tok else {}

    # 1. full tree
    url = "https://huggingface.co/api/datasets/%s/tree/%s?recursive=1&expand=true" % (
        args.repo_id,
        args.revision,
    )
    tree = []
    while url:
        r = httpx.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        tree.extend(r.json())
        link = r.headers.get("link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None

    files = [e for e in tree if e.get("type") == "file"]
    with_oid = [e for e in files if HEX64.match((e.get("lfs") or {}).get("oid", ""))]
    redacted = [e for e in files if set((e.get("lfs") or {}).get("oid", "")) == {"*"}]

    if not files:
        print("FAILED: tree is empty")
        return 2
    if redacted or not with_oid:
        print(
            "FAILED: digests are redacted (%d of %d), which means this request was "
            "not authenticated. Run 'hf auth login' first. Refusing to write a tree "
            "whose digests would make the manifest check vacuous." % (len(redacted), len(files))
        )
        return 3

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "remote_tree_pinned.json"), "w", encoding="utf-8") as fh:
        json.dump(tree, fh, ensure_ascii=False, indent=1)
    mat = [e for e in files if e["path"].startswith("mat/") and e["path"].lower().endswith(".mat")]
    with open(os.path.join(args.out_dir, "remote_mat_files.json"), "w", encoding="utf-8") as fh:
        json.dump(mat, fh, ensure_ascii=False, indent=1)

    total = sum(e["size"] for e in mat)
    print("credential_used        : %s" % bool(tok))
    print("tree entries           : %d" % len(tree))
    print("file entries           : %d" % len(files))
    print("entries with real oid  : %d" % len(with_oid))
    print("redacted entries       : %d" % len(redacted))
    print("mat/* files            : %d" % len(mat))
    print("mat/* total bytes      : %d" % total)
    print("WROTE remote_tree_pinned.json, remote_mat_files.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
