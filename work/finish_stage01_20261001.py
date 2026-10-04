"""接手阶段01的执行记录；所有日志写入新的接手目录，不覆盖历史结果。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "outputs" / "jeon2019_optics"
OUT = Path((ROOT / "work/接手阶段01当前目录.txt").read_text(encoding="utf-8"))
env = os.environ.copy()
env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
records = []


def run(name, args, cwd):
    """分别保存标准输出及真实返回码；失败不隐藏，也不删除失败记录。"""
    print("执行：", name, flush=True)
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "-B"] + args, cwd=cwd, env=env, capture_output=True)
    with (OUT / (name + ".log")).open("xb") as f:
        f.write(result.stdout + result.stderr)
    record = {"name": name, "command": [sys.executable, "-B"] + args,
              "cwd": str(cwd), "exit_code": result.returncode, "elapsed_s": time.perf_counter() - start}
    records.append(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if result.returncode:
        print((result.stdout + result.stderr).decode("utf-8", errors="replace")[-3000:], flush=True)


run("全部单元测试", ["-m", "unittest", "discover", "-s", "tests", "-v"], PROJECT)
run("配置开关核验", [str(PROJECT / "docs/verify_config_switches.py"),
                    "--report", str(OUT / "配置开关证据")], ROOT)
run("默认参数完整运行", [str(PROJECT / "main.py"), "--run-dir", str(OUT / "完整运行")], ROOT)
with (OUT / "接手执行记录.json").open("x", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print("接手验证输出：", OUT, flush=True)
