"""MiniMax H3 preparation, fail-fast checks and local audio/video inference."""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "model-lock.json"
GIB = 1024 ** 3


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def files_for(workflow):
    lock = read_json(LOCK)
    transformer = "transformer_ref/" if workflow == "ref2va" else "transformer/"
    return {k: v for k, v in lock["files"].items()
            if not k.startswith(("transformer/", "transformer_ref/")) or k.startswith(transformer)}


def verify_model(model_dir, workflow):
    bad = []
    for name, size in files_for(workflow).items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size:
            bad.append(name)
    if bad:
        raise ValueError("Missing/incomplete model files. Run download before GPU inference: " + ", ".join(bad[:8]))
    stamp = model_dir / "h3-download.json"
    if not stamp.is_file() or read_json(stamp).get("revision") != read_json(LOCK)["revision"]:
        raise ValueError("Model revision marker missing/mismatched. Rerun download to verify the pinned snapshot.")


def load_config(path):
    c = read_json(path)
    if c.get("workflow") not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError("workflow must be t2va/fl2va/ref2va")
    for key in ("height", "width", "num_frames", "num_inference_steps", "seed", "memory_reserve_gb"):
        if type(c.get(key)) is not int:
            raise ValueError(f"{key} must be an integer")
    if any(c[k] < 32 or c[k] % 32 for k in ("height", "width")):
        raise ValueError("height/width must be positive multiples of 32")
    aligned = max(5, math.ceil((c["num_frames"] - 5) / 17) * 17 + 5)
    if not 120 <= aligned <= 360:
        raise ValueError("Aligned frame count must be within 5–15 seconds at 24fps; use 124/243/345 frames")
    if c["num_frames"] != aligned:
        raise ValueError(f"Use an aligned num_frames value: {aligned} (17*n+5)")
    if c["num_inference_steps"] < 2 or c["seed"] < 0 or c["memory_reserve_gb"] < 1:
        raise ValueError("Invalid steps/seed/memory margin")
    if c.get("memory_mode") not in {"staged", "auto_offload"}:
        raise ValueError("memory_mode must be staged/auto_offload")
    if c["memory_mode"] == "staged" and c["workflow"] != "t2va":
        raise ValueError("staged currently supports text-to-video only; use auto_offload for media inputs")
    if "prompt_file" in c:
        c["prompt"] = (ROOT / c["prompt_file"]).read_text(encoding="utf-8").strip()
    if not isinstance(c.get("prompt"), str) or not c["prompt"].strip():
        raise ValueError("Provide a nonempty prompt or prompt_file")
    if c["workflow"] == "t2va" and any(c.get(k) for k in ("image", "last_image", "references")):
        raise ValueError("t2va does not accept reference/keyframe inputs")
    if c["workflow"] == "fl2va" and not any(c.get(k) for k in ("image", "last_image")):
        raise ValueError("fl2va requires image and/or last_image")
    for key in ("image", "last_image"):
        if c.get(key):
            f = (ROOT / c[key]).resolve()
            if not f.is_file():
                raise ValueError(f"Missing {key}: {f}")
            c[key] = str(f)
    if c["workflow"] == "ref2va":
        refs = c.get("references", [])
        counts = {k: 0 for k in ("image", "video", "audio")}
        if not 1 <= len(refs) <= 12:
            raise ValueError("ref2va requires 1–12 references")
        for ref in refs:
            if ref.get("type") not in counts:
                raise ValueError("Reference type must be image/video/audio")
            counts[ref["type"]] += 1
            f = (ROOT / ref["path"]).resolve()
            if not f.is_file():
                raise ValueError(f"Missing reference: {f}")
            ref["path"] = str(f)
        if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
            raise ValueError("Reference limits: 9 images, 3 videos, 3 audio clips")
        if not counts["image"] + counts["video"]:
            raise ValueError("Audio cannot be the only reference")
    name = c.get("output_name", "h3")
    if not name or Path(name).name != name or any(ch in name for ch in '/\\:'):
        raise ValueError("output_name must be a filename stem")
    return c


def plan(workflow):
    files = files_for(workflow)
    size = sum(files.values())
    print(json.dumps({"workflow": workflow, "files": len(files), "bytes": size,
                      "GiB": round(size / GIB, 2), "GB": round(size / 1e9, 2),
                      "disk_recommended_GiB": math.ceil(size / GIB) + 25,
                      "model": read_json(LOCK)["repo_id"], "revision": read_json(LOCK)["revision"]}, indent=2))


def download(model_dir, workflow):
    from huggingface_hub import snapshot_download
    model_dir.mkdir(parents=True, exist_ok=True)
    files = files_for(workflow)
    missing_bytes = sum(size for name, size in files.items()
                        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size != size)
    free = shutil.disk_usage(model_dir).free
    if free < missing_bytes + 5 * GIB:
        raise ValueError(f"Need {missing_bytes / GIB + 5:.1f} GiB free; filesystem reports {free / GIB:.1f}. Check Drive quota too.")
    lock = read_json(LOCK)
    snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"],
                      allow_patterns=list(files), local_dir=model_dir, max_workers=4,
                      token=os.environ.get("HF_TOKEN") or None)
    for name, size in files.items():
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size != size:
            raise ValueError(f"Download incomplete: {name}")
    (model_dir / "h3-download.json").write_text(json.dumps({
        "repo_id": lock["repo_id"], "revision": lock["revision"], "workflow": workflow,
        "verified_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"Verified pinned model: {model_dir}")


def check(model_dir, c):
    import psutil
    import torch
    import diffusers
    from diffusers import ModularPipeline, ComponentsManager
    from transformers import Qwen3VLForConditionalGeneration
    from diffusers.utils.export_utils import encode_video
    verify_model(model_dir, c["workflow"])
    if not torch.cuda.is_available():
        raise ValueError("No CUDA GPU. Select Colab G4 before check/run.")
    props = torch.cuda.get_device_properties(0)
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    vm = psutil.virtual_memory()
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__, "gpu": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "vram_total_GiB": round(total_vram / GIB, 1), "vram_free_GiB": round(free_vram / GIB, 1),
            "ram_total_GiB": round(vm.total / GIB, 1), "ram_available_GiB": round(vm.available / GIB, 1),
            "disk_free_GiB": round(shutil.disk_usage(model_dir).free / GIB, 1),
            "memory_mode": c["memory_mode"], "frames": c["num_frames"],
            "duration_seconds": c["num_frames"] / 24, "denoiser_evaluations": c["num_inference_steps"] - 1}
    print(json.dumps(info, indent=2), flush=True)
    if free_vram < 80 * GIB:
        raise ValueError("This BF16 profile requires at least 80 GiB free VRAM. Release other processes or select G4 96GB.")
    min_ram = 85 if c["memory_mode"] == "staged" else 150
    if vm.available < min_ram * GIB:
        raise ValueError(f"This mode needs at least {min_ram} GiB available host RAM (conservative guard). Select High RAM; for t2va use staged.")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported by the selected runtime")
    # Real allocation + GEMM catches missing sm_120 support before loading 140GB of weights.
    x = torch.ones((32, 32), dtype=torch.bfloat16, device="cuda")
    _ = x @ x
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import torch.nn.functional as F
    q = torch.zeros((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attention = F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    del x, _, q, attention
    torch.cuda.empty_cache()
    return info


def load_local(pipe, model_dir):
    import torch
    # The published modular index references the remote repository explicitly.
    # Override the repository for every component, retaining its subfolder.
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(model_dir),
                         local_files_only=True)
    missing = [name for name, value in pipe.components.items() if value is None]
    if missing:
        raise RuntimeError(f"Components failed to load: {missing}. Inspect preceding diffusers warnings.")


def run(model_dir, out_dir, c):
    # Set before importing Hugging Face libraries: GPU run must use the prepared snapshot.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    info = check(model_dir, c)
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = c.get("output_name", "h3") + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    video_path = out_dir / (stem + ".mp4")
    metadata = {"config": c, "environment": info, "model": read_json(LOCK)["revision"],
                "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    meta_path = out_dir / (stem + ".json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    kwargs = {k: c[k] for k in ("prompt", "height", "width", "num_frames", "num_inference_steps")}
    kwargs["generator"] = torch.Generator(device="cpu").manual_seed(c["seed"])
    kwargs["output"] = ["videos", "audio", "sampling_rate"]
    for key in ("image", "last_image"):
        if c.get(key):
            kwargs[key] = load_image(c[key])
    if c["workflow"] == "ref2va":
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference)
        classes = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
        kwargs["references"] = [classes[r["type"]].from_file(r["path"]) for r in c["references"]]
    try:
        with torch.inference_mode():
            if c["memory_mode"] == "staged":
                # Documented split-block API, adapted to one card: free the conditioner
                # completely before loading the denoiser. Only prompt state survives.
                workflow = ModularPipeline.from_pretrained(str(model_dir), local_files_only=True).blocks.get_workflow("t2va")
                encoder_blocks = workflow.sub_blocks.pop("text_encoder")
                encoder = encoder_blocks.init_pipeline(str(model_dir))
                print("Stage 1/2: loading text encoder only", flush=True)
                load_local(encoder, model_dir)
                encoder.to("cuda")
                state = encoder(prompt=c["prompt"])
                encoder.unload_components(list(encoder.components))
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
                manager = ComponentsManager()
                pipe = workflow.init_pipeline(str(model_dir), components_manager=manager)
                print("Stage 2/2: loading transformer + VAEs", flush=True)
                load_local(pipe, model_dir)
                manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=f"{c['memory_reserve_gb']}GB")
                results = pipe(state=state, **kwargs)
            else:
                manager = ComponentsManager()
                pipe = ModularPipeline.from_pretrained(str(model_dir), workflow=c["workflow"],
                                                        components_manager=manager, local_files_only=True)
                load_local(pipe, model_dir)
                manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=f"{c['memory_reserve_gb']}GB")
                results = pipe(**kwargs)
            encode_video(results["videos"][0], fps=24, output_path=str(video_path),
                         audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
        metadata.update(status="succeeded", video=str(video_path), elapsed_seconds=round(time.monotonic() - started, 2),
                        peak_cuda_allocated_GiB=round(torch.cuda.max_memory_allocated() / GIB, 2))
        print(f"Saved {video_path}", flush=True)
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=round(time.monotonic() - started, 2))
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "download", "verify", "check", "run", "validate-config"])
    p.add_argument("--workflow", choices=["t2va", "fl2va", "ref2va"], default="t2va")
    p.add_argument("--model-dir", type=Path, default=ROOT / "models/MiniMax-H3")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--config", type=Path, default=ROOT / "configs/smoke.json")
    a = p.parse_args()
    a.model_dir = a.model_dir.resolve()
    try:
        if a.action == "plan":
            plan(a.workflow)
        elif a.action == "download":
            download(a.model_dir, a.workflow)
        elif a.action == "verify":
            verify_model(a.model_dir, a.workflow)
            print("Model file sizes and revision marker OK")
        else:
            c = load_config(a.config)
            if a.action == "validate-config":
                print(f"Config OK: {c['workflow']}, {c['width']}x{c['height']}, {c['num_frames']}/24 seconds")
            elif a.action == "check":
                check(a.model_dir, c)
            else:
                run(a.model_dir, a.output_dir.resolve(), c)
    except (ValueError, OSError) as exc:
        p.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
