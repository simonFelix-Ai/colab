"""CPU-only API and MP4 muxing check. Does not download or load model weights."""
from pathlib import Path
import tempfile

import av
import numpy as np
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration
from diffusers import ComponentsManager, MiniMaxH3Blocks, ModularPipeline
from diffusers.utils.export_utils import encode_video


def main():
    blocks = MiniMaxH3Blocks()
    for mode in ("t2va", "fl2va", "ref2va"):
        workflow = blocks.get_workflow(mode)
        assert workflow is not None
    workflow = blocks.get_workflow("t2va")
    encoder = workflow.sub_blocks.pop("text_encoder").init_pipeline()
    rest = workflow.init_pipeline(components_manager=ComponentsManager())
    assert "text_encoder" in encoder.components
    assert "text_encoder" not in rest.components
    assert "transformer" in rest.components
    assert "transformer_ref" not in rest.components
    assert hasattr(encoder, "unload_components")
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "mux.mp4"
        frames = [Image.new("RGB", (64, 64), (20, 100, 180)) for _ in range(24)]
        encode_video(frames, fps=24, output_path=str(output),
                     audio=torch.zeros((2, 32000)), audio_sample_rate=32000)
        with av.open(str(output)) as container:
            assert any(s.type == "video" for s in container.streams)
            assert any(s.type == "audio" for s in container.streams)
    print(f"CPU checks passed: torch={torch.__version__}; H3 blocks/split API; MP4 video+audio encoding")


if __name__ == "__main__":
    main()
