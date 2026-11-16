"""Small CPU-only checks for expensive-runtime guardrails; no ML dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3", ROOT / "scripts/h3.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = h3.read_json(ROOT / "configs/smoke.json")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"

    def load(self, **updates):
        c = copy.deepcopy(self.config)
        c.update(updates)
        self.path.write_text(json.dumps(c))
        return h3.load_config(self.path)

    def test_checked_in_text_configs(self):
        for name in ("smoke", "quality"):
            c = h3.load_config(ROOT / f"configs/{name}.json")
            self.assertTrue(c["prompt"])
            self.assertEqual(c["num_frames"], 124)

    def test_invalid_canvas(self):
        with self.assertRaisesRegex(ValueError, "multiples"):
            self.load(width=961)

    def test_boolean_does_not_count_as_steps(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            self.load(num_inference_steps=True)

    def test_unaligned_frames(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            self.load(num_frames=125)

    def test_duration_rounding_cannot_exceed_limit(self):
        for frames in (119, 360, 362):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                self.load(num_frames=frames)

    def test_longer_aligned_durations(self):
        for frames in (243, 345):
            self.assertEqual(self.load(num_frames=frames)["num_frames"], frames)

    def test_image_mode_requires_file(self):
        with self.assertRaisesRegex(ValueError, "requires image"):
            self.load(workflow="fl2va", memory_mode="auto_offload")

    def test_staged_cannot_silently_ignore_images(self):
        with self.assertRaisesRegex(ValueError, "does not accept"):
            self.load(image="missing.png")
        with self.assertRaisesRegex(ValueError, "text-to-video only"):
            self.load(workflow="fl2va", image="missing.png")

    def test_audio_only_reference_rejected(self):
        audio = Path(self.tmp.name) / "audio.wav"
        audio.write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "only reference"):
            self.load(workflow="ref2va", memory_mode="auto_offload",
                      references=[{"type": "audio", "path": str(audio)}])

    def test_output_cannot_escape_directory(self):
        with self.assertRaisesRegex(ValueError, "filename stem"):
            self.load(output_name="../escape")


class ModelTests(unittest.TestCase):
    def test_no_duplicate_original_checkpoints_or_other_transformer(self):
        files = h3.files_for("t2va")
        self.assertTrue(any(k.startswith("transformer/") for k in files))
        self.assertFalse(any(k.startswith(("FL2VA/", "Ref2VA/", "transformer_ref/")) for k in files))
        self.assertEqual(files, h3.files_for("fl2va"))
        refs = h3.files_for("ref2va")
        self.assertTrue(any(k.startswith("transformer_ref/") for k in refs))
        self.assertFalse(any(k.startswith("transformer/") for k in refs))
        self.assertEqual(sum(files.values()), 144051160615)

    def test_manifest_and_notebooks_have_no_credentials(self):
        lock = h3.read_json(h3.LOCK)
        self.assertRegex(lock["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(lock["diffusers_revision"], r"^[0-9a-f]{40}$")
        for name, size in lock["files"].items():
            self.assertGreater(size, 0)
            self.assertNotIn("..", Path(name).parts)

    def test_missing_truncated_wrong_revision_and_complete_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            lock = folder / "lock.json"
            lock.write_text(json.dumps({"revision": "abc", "files": {"weights.bin": 4}}))
            with patch.object(h3, "LOCK", lock):
                with self.assertRaisesRegex(ValueError, "Missing/incomplete"):
                    h3.verify_model(folder, "t2va")
                (folder / "weights.bin").write_bytes(b"123")
                with self.assertRaisesRegex(ValueError, "Missing/incomplete"):
                    h3.verify_model(folder, "t2va")
                (folder / "weights.bin").write_bytes(b"1234")
                (folder / "h3-download.json").write_text('{"revision":"old"}')
                with self.assertRaisesRegex(ValueError, "revision"):
                    h3.verify_model(folder, "t2va")
                (folder / "h3-download.json").write_text('{"revision":"abc"}')
                h3.verify_model(folder, "t2va")


class NotebookTests(unittest.TestCase):
    def test_notebooks_are_reproducible_and_cells_compile(self):
        paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
        before = {path: path.read_bytes() for path in paths}
        subprocess.run([sys.executable, str(ROOT / "tools/build_notebooks.py")], check=True)
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertEqual(before[path], path.read_bytes())
            n = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(n["nbformat"], 4)
            ids = []
            for i, cell in enumerate(n["cells"]):
                ids.append(cell["id"])
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    compile("".join(cell["source"]), f"{path.name}:cell{i}", "exec")
            self.assertEqual(len(ids), len(set(ids)))

    def test_plan_needs_no_machine_learning_packages(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/h3.py"), "plan"],
                                text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)["files"], 62)


if __name__ == "__main__":
    unittest.main()
