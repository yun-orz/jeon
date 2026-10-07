# -*- coding: utf-8 -*-
"""按用户明确授权删除冗余下载产物，并留下可核对的删除记录。

删除范围（用户2026-10-07明确选择"删纯重复项，保留整包wheel"并允许删除旧部分下载原件）：

1. resume工作目录内 33 个 part_*.bin：拼接时整包后半段即由它们产生，属逐字节重复。
2. resume工作目录内 prefix.bin：旧部分下载的只读副本，原件另有其物。
3. 旧部分下载原件 gpu_cu121_20261006/torch-...whl：已被完整整包取代，续传不再需要。

保留：整包 torch wheel（已核验官方SHA，供本地离线重装）、sympy wheel、下载清单、
probe.json、install日志、续传状态、所有审核与清单文件、GPU虚拟环境、原CPU环境。

安全措施：逐个核对绝对路径必须位于预期目录内、文件名必须匹配预期模式、字节数必须
等于记录值；任一不符即中止，不删除。删除前记录每个文件的SHA256与字节数。
"""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESUME = ROOT / "work/dependencies/gpu_cu121_resume_m1_20261007"
OLD_DIR = ROOT / "work/dependencies/gpu_cu121_20261006"
OLD_PARTIAL = OLD_DIR / "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
AUDIT_DIR = ROOT / "outputs/执行_M1_GPU环境_20261007"
BLOCK_BYTES = 33554432          # 33 个分块，每块严格等于该长度


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1 << 22)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main():
    targets = []
    manifest = json.loads((RESUME / "download_manifest.json").read_text(encoding="utf-8"))
    expected = {}
    for item in manifest["blocks"]:
        path = RESUME / ("part_%03d.bin" % int(item["index"]))
        expected[path.resolve()] = int(item["bytes"])
    parts = sorted(RESUME.glob("part_*.bin"))
    if len(parts) != 33 or len(expected) != 33:
        raise SystemExit("分块数量不是33，拒绝删除：%d / %d" % (len(parts), len(expected)))
    for path in parts:
        # 末块是余数，长度由续传清单记录的字节范围决定，不假定恒为32MiB。
        want = expected.get(path.resolve())
        if want is None:
            raise SystemExit("分块未出现在续传清单中，拒绝删除：%s" % path.name)
        if path.stat().st_size != want:
            raise SystemExit("分块长度与清单不符，拒绝删除：%s（%d≠%d）"
                             % (path.name, path.stat().st_size, want))
        targets.append(path)
    prefix_copy = RESUME / "prefix.bin"
    if not prefix_copy.is_file():
        raise SystemExit("找不到 prefix.bin 副本，拒绝继续")
    targets.append(prefix_copy)
    if not OLD_PARTIAL.is_file():
        raise SystemExit("找不到旧部分下载原件，拒绝继续")
    targets.append(OLD_PARTIAL)

    # 路径与模式校验
    for path in targets:
        resolved = path.resolve()
        inside_resume = resolved.is_relative_to(RESUME.resolve())
        inside_old = resolved.is_relative_to(OLD_DIR.resolve())
        if not (inside_resume or inside_old):
            raise SystemExit("目标越出预期目录，拒绝删除：" + str(resolved))
    if not prefix_copy.resolve().is_relative_to(RESUME.resolve()):
        raise SystemExit("prefix.bin 路径异常，拒绝删除")

    records = []
    for path in targets:
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
                        "role": ("块文件（与整包后半段逐字节重复）" if path.name.startswith("part_")
                                 else "prefix.bin 副本（原件另有其物）" if path.name == "prefix.bin"
                                 else "旧部分下载原件（已被完整整包取代）")})
    total = sum(item["bytes"] for item in records)

    deleted = []
    for item in records:
        path = Path(item["path"])
        size_before = path.stat().st_size
        if size_before != item["bytes"]:
            raise SystemExit("删除前长度改变，中止：" + str(path))
        path.unlink()
        if path.exists():
            raise SystemExit("删除失败：" + str(path))
        deleted.append(item)

    record = {"schema": "jeon2019-m1-deletion-record-v1", "milestone": "M1",
              "deleted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "authorization": "用户2026-10-07明确同意：删纯重复项并保留整包wheel；并允许删除旧部分下载原件",
              "freed_bytes": total, "freed_gib": round(total / 1024 ** 3, 3),
              "deleted_count": len(deleted), "deleted": deleted,
              "kept": [str(RESUME / "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"),
                       str(RESUME / "sympy-1.13.1-py3-none-any.whl"),
                       str(RESUME / "download_manifest.json"), str(RESUME / "probe.json"),
                       str(RESUME / "resume_state.json"), str(RESUME / "install_result.json"),
                       str(RESUME / "install_log.txt")],
              "kept_reason": "整包wheel已核验官方SHA，是唯一能免联网重装Torch的本地来源；清单与日志是M1证据，不随二进制删除",
              "evidence_still_valid": "续传清单.json逐块记录字节范围与SHA256；下载SHA.json记录整包重算SHA与官方一致结论；M1判定不依赖这些文件继续存在",
              "irreversible_cost": "整包wheel删除后需重新联网下载2.3GiB才能重装Torch；本次保留wheel故该成本未发生",
              "recovery_commands": ["python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py download",
                                    "python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py install"],
              "note": "GPU虚拟环境work/environments/jeon_gpu_cu121已含安装后的Torch，运行不依赖被删文件；原CPU环境未受影响"}
    destination = AUDIT_DIR / "删除记录.json"
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已删除 %d 个文件，释放 %d 字节（%.3f GiB）" % (len(deleted), total, total / 1024 ** 3))
    print("删除记录：" + str(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
