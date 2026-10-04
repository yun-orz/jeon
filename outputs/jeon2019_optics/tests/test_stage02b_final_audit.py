# -*- coding: utf-8 -*-
"""复审剩余缺口：复用分析必须拒绝伪造功率，完成检查不能遗漏交付。"""
import json
import logging
from pathlib import Path
import shutil
import unittest
from unittest import mock
import contextlib
import io

import numpy as np
import main_stage02b as M
from optics.coordinates import make_grid
from tests._stage02b1_support import retained_evidence_dir


class TestFinalAudit(unittest.TestCase):
    def test_height_partial_and_report_failure_recorded(self):
        cfg = M.load_and_validate_config(M.PROJECT_ROOT / "config_stage02b.json")
        for inject in (False, True):
            target = retained_evidence_dir("final_audit_height_report") / "run"
            M.prepare_run_dir(target)
            logger = logging.getLogger("quiet_height_audit")
            if inject:
                with mock.patch.object(M, "build_report", side_effect=RuntimeError("报告写入反例")):
                    with self.assertRaisesRegex(RuntimeError, "报告写入反例"):
                        M.run_stage02b1(cfg, target, "height", logger)
                state = json.loads((target / "completion.json").read_text("utf-8"))
                self.assertEqual(state["status"], "failed")
            else:
                state = M.run_stage02b1(cfg, target, "height", logger)["state"]
                self.assertEqual(state["status"], "partial")
                self.assertTrue((target / "report_stage02b1.md").exists())
            self.assertEqual(state, json.loads((target / "checkpoint.json").read_text("utf-8")))

    def test_no_save_show_creates_figures_and_calls_show(self):
        # 使用Agg和真实计算/图形生命周期，仅替换弹窗，避免阻塞人工桌面。
        plt, info = M.configure_plotting(False)
        before = set(plt.get_fignums())
        with mock.patch.object(M, "configure_plotting", return_value=(plt, info)) as backend, \
                mock.patch.object(plt, "show") as show, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(M.main(["--no-save", "--show-plots"]), 0)
            backend.assert_called_once_with(True)
            show.assert_called_once()
        created = set(plt.get_fignums()) - before
        self.assertEqual(len(created), 7)
        for figure in created:
            plt.close(figure)

    def test_all_requires_analysis_and_delivery(self):
        checks = {name: {"status": "pass"} for name in M.required_checks_02b("all", True)}
        self.assertTrue(M.checks_pass_02b(checks, "all", True))
        for name in ("analysis_completed", "report_delivered", "required_artifacts_delivered"):
            altered = dict(checks)
            altered.pop(name)
            self.assertFalse(M.checks_pass_02b(altered, "all", True))

    def test_from_run_rejects_forged_power(self):
        # 使用实际送审固定场的复制件；旧源run绝不修改，测试证据不自动删除。
        cfg = M.load_and_validate_config(M.PROJECT_ROOT / "config_stage02b.json")
        src = M.PROJECT_ROOT / "results/stage02b1/run_20261002_163101"
        target = retained_evidence_dir("final_audit_forged_power")
        (target / "arrays").mkdir()
        for f in (src / "arrays").glob("*.npz"):
            shutil.copy2(f, target / "arrays" / f.name)
        for name in ("config_effective.json", "source_manifest.json"):
            shutil.copy2(src / name, target / name)
        psf = target / "arrays/psf_jeon_540nm.npz"
        with np.load(psf) as data:
            payload = {key: data[key] for key in data.files}
        payload["Pin"] *= 2
        payload["Pwindow"] *= 2
        np.savez_compressed(psf, **payload)
        with self.assertRaisesRegex(ValueError, "独立重算|重积分"):
            M.analyze_existing_run(target, cfg, make_grid(301, 1e-6), logging.getLogger("audit"))
