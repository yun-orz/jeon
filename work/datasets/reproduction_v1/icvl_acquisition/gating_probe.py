#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gating_probe.py -- prove/deny anonymous blob access for the pinned revision."""
import json
import os
import sys

import httpx

OUT = os.path.dirname(os.path.abspath(__file__))
SHA = "d2cf6714224029431cf4cec551ca6753ed59bc52"
BASE = "https://huggingface.co/datasets/ICVL-BGU/ICVL_HS_2016/resolve/%s/" % SHA

# utf-8-sig: these JSON files are written by PowerShell 5.1 Set-Content -Encoding UTF8,
# which emits a BOM.
tree = json.load(open(os.path.join(OUT, "remote_tree_pinned.json"), encoding="utf-8-sig"))
mats = [e["path"] for e in tree if e.get("type") == "file" and e["path"].startswith("mat/")]
print("mat files in pinned tree :", len(mats))
print("first mat path           :", mats[0])

# The task doc's dry-run template uses "mat/1.mat"; report it separately from the
# real per-file probes so a 401 caused by a NON-EXISTENT name is not mistaken for gating.
try:
    r0 = httpx.head(BASE + "1.mat", timeout=30)
    print("HEAD [template] 1.mat    :", r0.status_code,
          "(doc example name may not exist in the repo)")
except Exception as e:
    print("HEAD [template] 1.mat err:", type(e).__name__, e)

for label, rel in (("first", mats[0]), ("last", mats[-1])):
    url = BASE + rel
    try:
        r = httpx.head(url, timeout=30)
        print(f"HEAD [{label}] {rel}")
        print("   status         :", r.status_code)
        print("   x-linked-size  :", r.headers.get("x-linked-size"))
        print("   accept-ranges  :", r.headers.get("accept-ranges"))
        if r.status_code == 200:
            print("   -> ANONYMOUS ACCESS WORKS")
        elif r.status_code in (401, 403):
            print("   -> GATED: token required")
    except Exception as e:
        print(f"HEAD [{label}] error: {type(e).__name__}: {e}")

# Is there a local token at all?
try:
    from huggingface_hub import get_token

    tok = get_token()
    print("get_token()              :", "present" if tok else "None (not logged in)")
except Exception as e:
    print("get_token() error        :", type(e).__name__, e)

sys.exit(0)
