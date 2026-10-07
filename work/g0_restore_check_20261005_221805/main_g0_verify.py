# -*- coding: utf-8 -*-
"""只读核对 G0 冻结清单；可在 PyCharm 无参数运行，不产生或删除文件。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "baseline/g0_20261005/manifest.json"


def file_sha256(path):
    """分块读取，避免一次载入大型复场数组。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest, root=ROOT, code_only=False):
    """核对文件内容；清单与标签共同定义快照，不代表物理结论全部成立。"""
    if manifest.get("schema") != "jeon2019-g0-v1":
        raise ValueError("未知冻结清单格式")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("冻结清单为空或格式错误")
    root = Path(root).resolve()
    seen, failures, checked, skipped = set(), [], 0, 0
    for record in records:
        name = record["path"]
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or "\\" in name or ":" in name or name in seen):
            raise ValueError("非法或重复的冻结路径：" + name)
        seen.add(name)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("冻结路径越出项目：" + name)
        if record["scope"] not in {"code_document", "local_evidence"}:
            raise ValueError("未知文件范围：" + name)
        digest = record["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("非法 SHA256：" + name)
        if type(record["bytes"]) is not int or record["bytes"] < 0:
            raise ValueError("非法文件长度：" + name)
        if code_only and record["scope"] == "local_evidence":
            skipped += 1
            continue
        checked += 1
        if not path.is_file():
            failures.append("缺少文件：" + name)
        elif path.stat().st_size != record["bytes"] or file_sha256(path) != digest:
            failures.append("内容变化：" + name)
    return {"passed": not failures, "checked": checked, "skipped": skipped,
            "failures": failures, "mode": "code_only" if code_only else "full_local"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true",
                        help="仅核对代码文档；不验证本地数组与运行证据")
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        report = verify_manifest(manifest, code_only=args.code_only)
        print("G0冻结核对：" + ("通过" if report["passed"] else "未通过"))
        print(f"模式={report['mode']}，核对={report['checked']}，跳过={report['skipped']}")
        for message in report["failures"][:12]:
            print(message)
        if len(report["failures"]) > 12:
            print(f"另有 {len(report['failures']) - 12} 项不一致")
        if args.code_only:
            print("代码文档通过不能证明本地数值证据齐全。")
        return 0 if report["passed"] else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("G0冻结核对失败：" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
