# -*- coding: utf-8 -*-
"""运行M1 check并写入新M1结果目录；安装前后各执行一次。

用法（两种解释器都可运行）：

    python -B m1_check.py --stage pre_install
    work/environments/jeon_gpu_cu121/Scripts/python.exe -B m1_check.py --stage post_install

每次执行都新开一个 run 目录，不覆盖历史；check 本身只读，不安装、不下载。
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m1_environment  # noqa: E402


def new_run(stage):
    destination = m1_environment.RESULTS / "m1" / ("run_%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), stage))
    index = 1
    while destination.exists():
        destination = m1_environment.RESULTS / "m1" / ("run_%s_%s_r%d" % (time.strftime("%Y%m%d_%H%M%S"), stage, index))
        index += 1
    destination.mkdir(parents=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="pre_install", help="标记本次check的阶段名，只影响目录名")
    parser.add_argument("--run-dir", default=None, help="复用已有run目录，避免重复创建")
    arguments = parser.parse_args()
    run = Path(arguments.run_dir) if arguments.run_dir else new_run(arguments.stage)
    run.mkdir(parents=True, exist_ok=True)
    report = m1_environment.check(persist=False)
    report["stage"] = arguments.stage
    report["run_dir"] = str(run)
    report["operator_command"] = {"executable": sys.executable, "argv": sys.argv}
    report["disk_snapshot"] = {"project_drive_free_bytes": m1_environment.disk_free(m1_environment.ROOT),
                               "temperature_free_bytes": m1_environment.disk_free(
                                   Path(shutil.os.environ.get("TEMP", "C:/Windows/Temp")))}
    destination = run / ("m1_check_%s.json" % arguments.stage)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "stage": report["stage"],
                      "gpu_environment_ready": report["gpu_environment_ready"],
                      "torch_from_environment": report["torch_from_environment"],
                      "space_budget_passed": report["space"]["budget"]["passed"],
                      "written_to": str(destination)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
