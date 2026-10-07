# -*- coding: utf-8 -*-
"""M1 续传器与空间核算测试；只使用本地回环样例服务与小型临时文件。

覆盖：正常分块206、已有前缀续传、200替代206、长度不符重试、
真实进程被杀后的恢复、清单丢失后的磁盘复核、损坏块重下、剩余空间不足停止。
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m1_fixture_server import start_fixture  # noqa: E402
from m1_resume import Resumer, ResumeError, plan_blocks, sha256_file  # noqa: E402
import m1_environment  # noqa: E402

PAYLOAD = bytes(range(256)) * 2048          # 512 KiB
BLOCK = 64 * 1024


class ResumeCase(unittest.TestCase):
    """公共脚手架：每个用例一个临时目录与一个本地样例服务。"""

    def setUp(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="m1_resume_"))
        self.fixture, self.server, self.url = start_fixture(PAYLOAD)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)

    def resumer(self, name="sample.bin", prefix=None, **kwargs):
        options = dict(url=self.url, total_bytes=len(PAYLOAD), sha256=self.fixture.sha256(),
                       destination=self.temporary / name, block_bytes=BLOCK, logger=lambda *a: None,
                       backoff=0.01, prefix=prefix)
        options.update(kwargs)
        return Resumer(**options)


class TestResumer(ResumeCase):
    def test_block_plan_covers_every_byte(self):
        self.assertEqual(plan_blocks(0, 10, 4), [(0, 0, 3), (1, 4, 7), (2, 8, 9)])
        self.assertEqual(plan_blocks(10, 10, 4), [])
        with self.assertRaises(ResumeError):
            plan_blocks(11, 10, 4)

    def test_full_download_uses_206_and_passes_whole_sha(self):
        engine = self.resumer()
        manifest = engine.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertEqual(sha256_file(engine.destination), self.fixture.sha256())
        self.assertEqual(manifest["prefix_bytes"], 0)
        self.assertEqual(len(manifest["blocks"]), 8)
        self.assertTrue(all(item["status"] == 206 for item in self.fixture.requests if item["method"] == "GET"))
        self.assertEqual([item["range"] for item in self.fixture.requests if item["method"] == "GET"],
                         ["bytes=%d-%d" % (start, end) for _, start, end in plan_blocks(0, len(PAYLOAD), BLOCK)])

    def test_existing_prefix_is_copied_and_never_modified(self):
        prefix = self.temporary / "partial.whl"
        prefix.write_bytes(PAYLOAD[:BLOCK + 123])
        before = sha256_file(prefix)
        engine = self.resumer(prefix=prefix)
        manifest = engine.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertEqual(manifest["prefix_bytes"], BLOCK + 123)
        self.assertEqual(engine.prefix_path.read_bytes(), PAYLOAD[:BLOCK + 123])
        self.assertEqual(sha256_file(prefix), before, "原前缀被改动")
        ranges = [item["range"] for item in self.fixture.requests if item["method"] == "GET"]
        self.assertEqual(ranges[0], "bytes=%d-%d" % (BLOCK + 123, 2 * BLOCK + 122))

    def test_server_ignoring_range_is_accepted_only_as_one_whole_block(self):
        self.fixture.ignore_range = True
        prefix = self.temporary / "partial.whl"
        prefix.write_bytes(PAYLOAD[:1000])
        engine = self.resumer(prefix=prefix)
        manifest = engine.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertEqual(len(manifest["blocks"]), 1)
        self.assertEqual(manifest["blocks"][0]["range"], "bytes 1000-%d/%d" % (len(PAYLOAD) - 1, len(PAYLOAD)))
        self.assertTrue(any(item["kind"] == "server_ignored_range" for item in manifest["events"]))

    def test_permanent_ignored_range_with_short_body_fails_without_fake_block(self):
        """服务端返回200但正文被截断：必须失败，且状态文件不得登记任何完成块。"""
        self.fixture.ignore_range = True
        self.fixture.truncate = True
        engine = self.resumer(block_bytes=BLOCK, attempts=2)
        with self.assertRaises(ResumeError):
            engine.run()
        self.assertFalse(engine.destination.exists(), "失败时不得生成整包")
        state = json.loads(engine.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["blocks"], [], "未核验的块不得登记为完成")
        self.assertEqual(state["status"], "block_failed")
        self.assertFalse(any(item["kind"] == "whole_sha_passed" for item in engine.events))
        self.assertTrue(any(item["kind"] == "block_attempt_failed" for item in engine.events),
                        "长度不符必须留下失败记录")

    def test_ignored_range_with_prefix_accepted_by_skipping_prefix(self):
        """200 整包响应配合已有前缀：跳过前缀字节，仍拼出正确整包。"""
        self.fixture.ignore_range = True
        prefix = self.temporary / "partial.whl"
        prefix.write_bytes(PAYLOAD[:1000])
        engine = self.resumer(prefix=prefix)
        manifest = engine.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertEqual(sha256_file(engine.destination), self.fixture.sha256())
        self.assertEqual(len(manifest["blocks"]), 1)
        self.assertTrue(any(item["kind"] == "server_ignored_range" for item in manifest["events"]))

    def test_length_mismatch_retries_then_succeeds(self):
        self.fixture.truncate_blocks = 2
        engine = self.resumer(attempts=3)
        manifest = engine.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertEqual(sha256_file(engine.destination), self.fixture.sha256())
        failed = [item for item in manifest["events"] if item["kind"] == "block_attempt_failed"]
        self.assertGreaterEqual(len(failed), 2, "长度不符的失败尝试必须全部记录：%s" % failed)
        self.assertTrue(all(any(word in item["error"] for word in ["IncompleteRead", "长度不符", "响应正文写入"])
                            for item in failed), failed)

    def test_corrupted_block_is_redownloaded(self):
        """已登记SHA的块被破坏后必须重下，不能凭大小相同蒙混过关。"""
        engine = self.resumer()
        engine.run()
        target = engine.block_path(3)
        target.write_bytes(b"0" * target.stat().st_size)
        engine.destination.unlink()                      # 只保留分块，强制重新拼接
        self.fixture.requests.clear()
        second = self.resumer(name="sample.bin")         # 复用同一目录与状态清单
        manifest = second.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertTrue(any(item["kind"] == "block_corrupt" for item in manifest["events"]))
        self.assertEqual(sha256_file(second.destination), self.fixture.sha256())
        fetched = [item for item in self.fixture.requests if item["method"] == "GET"]
        self.assertEqual([item["range"] for item in fetched], ["bytes=196608-262143"],
                         "只应重新下载被破坏的第3块：%s" % fetched)
        self.assertTrue(all(item["status"] == 206 for item in fetched))

    def test_state_file_removed_still_recovers_from_disk(self):
        engine = self.resumer()
        engine.run()
        engine.destination.unlink()
        engine.state_path.unlink()
        fresh = self.resumer()
        self.fixture.requests.clear()
        manifest = fresh.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertTrue(all(item["kind"] != "block_attempt_failed" for item in manifest["events"]))
        self.assertTrue(any(item["kind"] == "orphan_block_adopted" for item in manifest["events"]),
                        "磁盘上已完成的块必须被复核后复用")
        self.assertEqual([item for item in self.fixture.requests if item["method"] == "GET"], [],
                         "清单丢失但块完好时不应重新下载")

    def test_second_run_short_circuits_when_whole_package_present(self):
        engine = self.resumer()
        engine.run()
        self.fixture.requests.clear()
        manifest = self.resumer().run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertTrue(any(item["kind"] == "already_complete" for item in manifest["events"]))
        self.assertEqual([item for item in self.fixture.requests if item["method"] == "GET"], [])

    def test_probe_detects_wrong_frozen_total(self):
        wrong = Resumer(url=self.url, total_bytes=len(PAYLOAD) + 1, sha256=self.fixture.sha256(),
                        destination=self.temporary / "probe.bin", block_bytes=BLOCK, logger=lambda *a: None)
        result = wrong.probe()
        self.assertFalse(result["passed"])
        self.assertTrue(any("不一致" in item for item in result["problems"]))
        right = self.resumer(name="probe_ok.bin")
        self.assertTrue(right.probe()["passed"])


class TestHardKillRecovery(ResumeCase):
    """真实进程被杀：用子进程下载中途终止，再用同一目录重启续传。"""

    def test_process_killed_midway_then_resumed(self):
        workdir = self.temporary / "killed"
        script = Path(__file__).resolve().with_name("m1_resume_cli_fixture.py")
        first = subprocess.Popen([sys.executable, "-B", str(script), "--url", self.url,
                                  "--total", str(len(PAYLOAD)), "--sha256", self.fixture.sha256(),
                                  "--destination", str(workdir / "sample.bin"), "--block-bytes", str(BLOCK),
                                  "--kill-after-blocks", "2"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = first.communicate(timeout=180)[0]
        self.assertEqual(first.returncode, 0, output)
        parts = sorted(workdir.glob("part_*.bin"))
        self.assertEqual(len(parts), 2, "被终止前应已完成两块：%s" % output)
        self.assertFalse((workdir / "sample.bin").exists())
        done = {json.loads(workdir.joinpath("resume_state.json").read_text(encoding="utf-8"))["blocks"][i]["index"]
                for i in range(2)}
        self.fixture.requests.clear()
        engine = Resumer(url=self.url, total_bytes=len(PAYLOAD), sha256=self.fixture.sha256(),
                         destination=workdir / "sample.bin", block_bytes=BLOCK, logger=lambda *a: None,
                         backoff=0.01)
        manifest = engine.run()
        self.assertTrue(manifest["whole_sha256_passed"])
        self.assertEqual(sha256_file(workdir / "sample.bin"), self.fixture.sha256())
        requested = [item["range"] for item in self.fixture.requests if item["method"] == "GET"]
        self.assertEqual(len(requested), len(plan_blocks(0, len(PAYLOAD), BLOCK)) - len(done),
                         "只应下载未完成块：%s" % requested)
        for index in done:
            self.assertTrue(engine.block_path(index).is_file())


class TestSpaceBudget(unittest.TestCase):
    def test_insufficient_space_stops_with_requirement(self):
        result = m1_environment.space_budget(free_bytes=10 ** 9, remaining_download=2 * 10 ** 9,
                                             assembled_copy=2 * 10 ** 9, install_estimate=4 * 10 ** 9)
        self.assertFalse(result["passed"])
        self.assertGreater(result["shortfall_bytes"], 0)
        self.assertTrue(result["stop_reason"])

    def test_sufficient_space_passes_with_margin(self):
        result = m1_environment.space_budget(free_bytes=100 * 10 ** 9, remaining_download=1.1 * 10 ** 9,
                                             assembled_copy=2.5 * 10 ** 9, install_estimate=4 * 10 ** 9)
        self.assertTrue(result["passed"])
        self.assertGreater(result["headroom_bytes"], 0)

    def test_peak_is_sum_of_components(self):
        result = m1_environment.space_budget(free_bytes=100 * 10 ** 9, remaining_download=1.0,
                                             assembled_copy=2.0, install_estimate=3.0, audit_reserve=4.0)
        self.assertEqual(result["peak_bytes"], 10.0)
        self.assertEqual(result["components"]["remaining_download"], 1.0)

    def test_existing_prefix_reduces_remaining_download(self):
        result = m1_environment.space_budget(free_bytes=1, remaining_download=2, assembled_copy=3,
                                             install_estimate=4, audit_reserve=5)
        self.assertEqual(result["components"]["assembled_copy"], 3)
        self.assertEqual(result["peak_bytes"], 14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
