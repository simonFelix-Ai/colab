"""Cache Linux wheels on CPU; restore an isolated environment on a fresh GPU VM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements-colab.txt"


def call(args):
    subprocess.run([str(x) for x in args], check=True)


def fingerprint():
    return {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "system": platform.system(), "machine": platform.machine(),
            "requirements_sha256": hashlib.sha256(REQ.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["cache", "install"])
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--venv", type=Path, default=ROOT / ".venv")
    a = p.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        p.error("Run this in Colab Linux x86_64, not on your Windows computer.")
    if sys.version_info[:2] != (3, 12):
        p.error("This environment targets Python 3.12. Choose Colab Python 3.12 or revise and recache requirements.")
    a.cache.mkdir(parents=True, exist_ok=True)
    marker = a.cache / "environment.json"
    wheels = a.cache / "wheels"
    if a.action == "cache":
        marker.unlink(missing_ok=True)
        wheels.mkdir(exist_ok=True)
        # A temporary clean venv prevents packages in the CPU runtime influencing resolution.
        with tempfile.TemporaryDirectory(prefix="h3-build-") as tmp:
            venv.EnvBuilder(with_pip=True).create(tmp)
            py = Path(tmp) / "bin/python"
            call([py, "-m", "pip", "install", "--upgrade", "pip"])
            call([py, "-m", "pip", "wheel", "-r", REQ, "--wheel-dir", wheels,
                  "--prefer-binary"])
            # Resolve an exact wheel set offline; this is the lock actually restored on GPU.
            offline_req = Path(tmp) / "offline-requirements.txt"
            offline_req.write_text("\n".join(
                "diffusers" if line.startswith("diffusers @") else line
                for line in REQ.read_text().splitlines()
                if line and not line.startswith(("#", "--"))) + "\n")
            call([py, "-m", "pip", "install", "--no-index", "--find-links", wheels, "-r", offline_req])
            call([py, "-m", "pip", "check"])
            call([py, ROOT / "scripts/validate_runtime.py"])
            freeze = subprocess.check_output([str(py), "-m", "pip", "list", "--format=json"], text=True)
            locked = [f"{x['name']}=={x['version']}" for x in json.loads(freeze)
                      if x['name'].lower() not in {"pip", "setuptools", "wheel"}]
            (a.cache / "resolved.txt").write_text("\n".join(locked) + "\n", encoding="utf-8")
        marker.write_text(json.dumps(fingerprint(), indent=2), encoding="utf-8")
        print("Environment cached. Keep this directory on Drive. No GPU was required.")
    else:
        if not marker.is_file() or json.loads(marker.read_text()) != fingerprint():
            p.error("Wheel cache missing or Python/platform/requirements changed. Run CPU cache stage again.")
        venv.EnvBuilder(with_pip=True).create(a.venv)
        py = a.venv / "bin/python"
        call([py, "-m", "pip", "install", "--no-index", "--find-links", wheels,
              "-r", a.cache / "resolved.txt"])
        call([py, "-m", "pip", "check"])
        print(f"Ready: {py}")


if __name__ == "__main__":
    main()
