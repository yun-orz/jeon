"""保护清单的负路径核对；只读现有小文件，不产生或清理fixture。"""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import HERE, ROOT, SCHEMA, protocol_check, safe_path, sha256, validate_records, verify_manifest


class TestProtection(unittest.TestCase):
    def manifest(self):
        path = HERE / "__init__.py"
        return {"schema": SCHEMA, "files": [{"path": path.relative_to(ROOT).as_posix(),
                  "bytes": path.stat().st_size, "sha256": sha256(path), "mode": "immutable"}]}

    def test_valid_content(self):
        self.assertTrue(verify_manifest(self.manifest())["passed"])

    def test_bad_sha_rejected(self):
        manifest = self.manifest()
        manifest["files"][0]["sha256"] = "0" * 64
        self.assertFalse(verify_manifest(manifest)["passed"])

    def test_bad_size_rejected(self):
        manifest = self.manifest()
        manifest["files"][0]["bytes"] += 1
        self.assertFalse(verify_manifest(manifest)["passed"])

    def test_missing_rejected(self):
        manifest = self.manifest()
        manifest["files"][0]["path"] = "outputs/jeon2019_optics/reproduction_v1/nonexistent_m0_fixture"
        self.assertFalse(verify_manifest(manifest)["passed"])

    def test_duplicate_rejected(self):
        manifest = self.manifest()
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
        with self.assertRaises(ValueError):
            validate_records(manifest)

    @unittest.skipUnless(sys.platform == "win32", "大小写路径别名仅在Windows检查")
    def test_windows_case_alias_rejected(self):
        manifest = self.manifest()
        other = copy.deepcopy(manifest["files"][0])
        other["path"] = other["path"].upper()
        manifest["files"].append(other)
        with self.assertRaises(ValueError):
            validate_records(manifest)

    def test_path_traversal_rejected(self):
        for value in ["../README.md", "/README.md", "D:/README.md", "outputs\\README.md", "./README.md", "outputs//x", ""]:
            with self.subTest(path=value), self.assertRaises(ValueError):
                safe_path(value)

    def test_bool_size_and_unknown_mode_rejected(self):
        for key, value in [("bytes", True), ("mode", "ignore_changes"), ("sha256", "BAD")]:
            manifest = self.manifest()
            manifest["files"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_records(manifest)

    def test_append_only_limited_to_status(self):
        manifest = self.manifest()
        manifest["files"][0]["mode"] = "append_only"
        with self.assertRaises(ValueError):
            validate_records(manifest)

    def test_append_only_prefix_accepts_suffix(self):
        path = ROOT / "CURRENT_STATUS.md"
        count = min(100, path.stat().st_size)
        manifest = {"schema": SCHEMA, "files": [{"path": "CURRENT_STATUS.md", "mode": "append_only",
                                                "bytes": count, "sha256": sha256(path, count)}]}
        self.assertTrue(verify_manifest(manifest)["passed"])
        manifest["files"][0]["sha256"] = "0" * 64
        self.assertFalse(verify_manifest(manifest)["passed"])

    def test_append_only_cannot_accept_truncation(self):
        path = ROOT / "CURRENT_STATUS.md"
        manifest = {"schema": SCHEMA, "files": [{"path": "CURRENT_STATUS.md", "mode": "append_only",
                        "bytes": path.stat().st_size + 1, "sha256": sha256(path)}]}
        self.assertFalse(verify_manifest(manifest)["passed"])

    def test_empty_and_bad_schema_rejected(self):
        for value in [{"schema": SCHEMA, "files": []}, {"schema": "other", "files": [1]}]:
            with self.assertRaises(ValueError):
                validate_records(value)

    def test_protocol_sources_and_tasks(self):
        result = protocol_check()
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["tasks"], 8)

    def test_pending_command_cannot_claim_success(self):
        import main
        with patch.object(sys, "argv", ["main.py", "train"]):
            self.assertEqual(main.main(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
