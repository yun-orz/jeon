# -*- coding: utf-8 -*-
"""阶段 02B-1 工作区清理：把冗余项**删除到回收站**（可还原）。

用途与边界
----------
* 本脚本只处理 `_cleanup_manifest.json` 中**显式列出**的绝对路径；
  不使用任何通配符扩展，绝不递归扫描后“顺手”删除未列出的东西。
* 删除方式为**送到回收站**，不是永久删除；用户可在回收站还原。
* 需要写权限（沙箱完全权限）才能运行；普通 workspace-write 下会被拒绝。

安全闸门（任一不通过即中止，不做任何删除）
------------------------------------------
1. 每个目标必须位于允许的根目录之下（`outputs/jeon2019_optics` 或 `work`）；
2. 每个目标必须真实存在；
3. 目标集合中不允许出现“保留清单”里的路径；
4. 目标不允许是允许根目录本身或其祖先。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ALLOWED_ROOTS = [
    Path(r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics"),
    Path(r"D:\PyCharmProjects\Jeon2019\work"),
]
MANIFEST = Path(r"D:\PyCharmProjects\Jeon2019\_cleanup_manifest.json")


def under_allowed(path: Path) -> bool:
    rp = path.resolve()
    for root in ALLOWED_ROOTS:
        rr = root.resolve()
        if rp == rr:
            return False                      # 不允许删根目录本身
        if rr in rp.parents:
            return True
    return False


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正执行；缺省只做预演")
    args = ap.parse_args()

    plan = json.loads(MANIFEST.read_text(encoding="utf-8"))
    targets = [Path(p) for p in plan["delete"]]
    keeps = {Path(p).resolve() for p in plan.get("keep", [])}

    # ---------------- 安全闸门 ----------------
    problems = []
    for t in targets:
        if not under_allowed(t):
            problems.append("越界（不在允许根目录下）：%s" % t)
        elif not t.exists():
            problems.append("不存在：%s" % t)
        elif t.resolve() in keeps:
            problems.append("同时出现在保留清单：%s" % t)
    if problems:
        print("安全闸门不通过，已中止，未删除任何内容：")
        for p in problems:
            print("  -", p)
        return 2

    # ---------------- 送回收站 ----------------
    # 说明：送回收站要用 Microsoft.VisualBasic 的 FileSystem.DeleteDirectory +
    # RecycleOption.SendToRecycleBin（.NET 的 Directory.Delete 是永久删除）。
    # 该程序集只在实际执行时才加载，预演分支不依赖它。
    if args.apply:
        import clr                                                     # noqa: F401
        clr.AddReference("Microsoft.VisualBasic")
        from Microsoft.VisualBasic.FileIO import FileSystem, UIOption, RecycleOption

    total_files = total_bytes = 0
    failed = []
    for t in targets:
        try:
            for f in t.rglob("*"):
                if f.is_file():
                    total_files += 1
                    total_bytes += f.stat().st_size
        except OSError:
            pass
        if not args.apply:
            print("  [预演] 将送回收站：%s" % t)
            continue
        try:
            if t.is_dir():
                FileSystem.DeleteDirectory(
                    str(t), UIOption.OnlyErrorDialogs, RecycleOption.SendToRecycleBin)
            else:
                FileSystem.DeleteFile(
                    str(t), UIOption.OnlyErrorDialogs, RecycleOption.SendToRecycleBin)
            print("  已送回收站：%s" % t)
        except Exception as exc:                                     # noqa: BLE001
            failed.append("%s -> %s: %s" % (t, type(exc).__name__, exc))

    print()
    print("目标数：%d；涉及文件 %d 个，约 %.1f MB"
          % (len(targets), total_files, total_bytes / 2 ** 20))
    if args.apply:
        print("成功：%d；失败：%d" % (len(targets) - len(failed), len(failed)))
    else:
        print("（预演模式，未删除任何内容；加 --apply 才真正执行）")
    for f in failed:
        print("  失败:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
