# -*- coding: utf-8 -*-
"""硬杀测试用的最小续传入口：完成指定数量分块后自杀，模拟进程中断。

只用于本地回环样例服务，不访问外网。
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m1_resume import Resumer, plan_blocks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--block-bytes", type=int, default=64 * 1024)
    parser.add_argument("--kill-after-blocks", type=int, default=0)
    args = parser.parse_args()

    engine = Resumer(url=args.url, total_bytes=args.total, sha256=args.sha256,
                     destination=Path(args.destination), block_bytes=args.block_bytes,
                     attempts=2, logger=lambda *a: None, backoff=0.01)
    engine.directory.mkdir(parents=True, exist_ok=True)
    engine.load_state()
    engine.ensure_prefix()
    blocks = plan_blocks(engine.offset, engine.total_bytes, engine.block_bytes)
    pending, _ = engine.verify_blocks(blocks)
    engine.save_state("downloading")
    for count, (index, start, end) in enumerate(pending, start=1):
        record = engine._download_block(index, start, end)
        engine.records.append(record)
        engine.records.sort(key=lambda item: int(item["index"]))
        engine.save_state("downloading")
        print("完成块%d" % index, flush=True)
        if args.kill_after_blocks and count >= args.kill_after_blocks:
            print("模拟进程被终止", flush=True)
            os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
