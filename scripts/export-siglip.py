"""Convert SigLIP 2 SO400M NaFlex to a fixed-256-patch OpenVINO package.

Run this on a developer machine. End users download the resulting IR and never
install PyTorch. The wrapper externalizes resized position embeddings because
Transformers itemizes ``spatial_shapes`` during tracing, which otherwise freezes
the one aspect ratio used by the conversion example.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


UPSTREAM = "google/siglip2-so400m-patch16-naflex"
UPSTREAM_REVISION = "cc24074f717b612951c2dead130904ab9b65a81e"
DEFAULT_OUTPUT = Path("release-models/siglip2-so400m-patch16-naflex-openvino-fp16")
MAX_PATCHES = 256


class VisionForOpenVINO(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model
        self.num_heads = self.vision.config.num_attention_heads

    def forward(self, pixel_values, pixel_attention_mask, position_embeddings):
        target_dtype = self.vision.embeddings.patch_embedding.weight.dtype
        hidden = self.vision.embeddings.patch_embedding(pixel_values.to(target_dtype))
        hidden = hidden + position_embeddings.to(target_dtype)
        # Additive key-padding mask: zero for real patches, large negative for padding.
        key_mask = (1.0 - pixel_attention_mask.to(target_dtype))[:, None, None, :] * -10000.0
        encoded = self.vision.encoder(inputs_embeds=hidden, attention_mask=key_mask).last_hidden_state
        encoded = self.vision.post_layernorm(encoded)

        # Spell out SigLIP2's attention-pooling head to avoid the generic dynamic
        # mask builder while preserving the original trained weights.
        head = self.vision.head
        probe = head.probe.repeat(encoded.shape[0], 1, 1)
        pool_mask = key_mask.repeat(1, self.num_heads, 1, 1).reshape(-1, 1, encoded.shape[1])
        pooled = head.attention(probe, encoded, encoded, attn_mask=pool_mask, need_weights=False)[0]
        residual = pooled
        pooled = residual + head.mlp(head.layernorm(pooled))
        return pooled[:, 0]


class TextForOpenVINO(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text = model.text_model

    def forward(self, input_ids, attention_mask):
        return self.text(input_ids=input_ids, attention_mask=attention_mask).pooler_output


def set_names(model: ov.Model, names: list[str]) -> None:
    for port, name in zip(model.inputs, names, strict=True):
        port.get_tensor().set_names({name})
    model.output(0).get_tensor().set_names({"embedding"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--upstream", default=UPSTREAM)
    parser.add_argument("--revision", default=UPSTREAM_REVISION)
    parser.add_argument("--force", action="store_true", help="replace an existing generated output folder")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise SystemExit(f"Output is not empty: {output}\nUse --force to replace generated files.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    print("1/5 Downloading the pinned upstream model (developer machine only)…")
    model = AutoModel.from_pretrained(
        args.upstream,
        revision=args.revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    model.config.vision_config._attn_implementation = "eager"
    model.config.text_config._attn_implementation = "eager"
    processor = AutoProcessor.from_pretrained(args.upstream, revision=args.revision, use_fast=False)
    processor.save_pretrained(output)

    print("2/5 Exporting the 256-patch NaFlex vision encoder…")
    sample = processor(images=[Image.new("RGB", (640, 360))], return_tensors="pt")
    base_positions = model.vision_model.embeddings.position_embedding.weight.detach().cpu()
    side = int(base_positions.shape[0] ** 0.5)
    base_positions = base_positions.reshape(side, side, -1)
    np.save(output / "position_embedding.npy", base_positions.numpy().astype(np.float16))
    resized_positions = model.vision_model.embeddings.resize_positional_embeddings(
        base_positions,
        sample["spatial_shapes"],
        max_length=MAX_PATCHES,
    )
    vision_ir = ov.convert_model(
        VisionForOpenVINO(model),
        example_input=(sample["pixel_values"], sample["pixel_attention_mask"], resized_positions),
    )
    set_names(vision_ir, ["pixel_values", "pixel_attention_mask", "position_embeddings"])
    ov.save_model(vision_ir, output / "vision_model.xml", compress_to_fp16=True)

    print("3/5 Exporting the multilingual text encoder…")
    text_inputs = processor(
        text=["一張桌上的紅色筆電"],
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(text_inputs["input_ids"])
    text_ir = ov.convert_model(
        TextForOpenVINO(model),
        example_input=(text_inputs["input_ids"], attention_mask),
    )
    set_names(text_ir, ["input_ids", "attention_mask"])
    ov.save_model(text_ir, output / "text_model.xml", compress_to_fp16=True)

    print("4/5 Writing package metadata…")
    metadata = {
        "format": "visual-search-openvino-naflex-v1",
        "upstream": args.upstream,
        "upstream_revision": args.revision,
        "precision": "FP16",
        "patch_size": 16,
        "max_num_patches": MAX_PATCHES,
        "embedding_dimensions": int(model.config.vision_config.hidden_size),
        "position_embeddings_externalized": True,
    }
    (output / "openvino_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("MODEL_CARD.md"), output / "README.md")

    print("5/5 Verifying OpenVINO on CPU…")
    # Import from source checkout without requiring an installed wheel.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from visual_search.engines import OpenVinoSiglipEngine
    engine = OpenVinoSiglipEngine(output, device="CPU")
    engine.smoke_test()
    print("Validating portrait, landscape, and square embeddings against PyTorch…")
    for size in ((360, 960), (960, 360), (640, 640)):
        image = Image.new("RGB", size, (70, 130, 190))
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            reference_output = model.get_image_features(**inputs)
            reference_tensor = getattr(reference_output, "pooler_output", reference_output)
            reference = reference_tensor.cpu().numpy().reshape(-1)
        converted = engine._image_from_pil(image).reshape(-1)
        cosine = float(np.dot(reference, converted) / (np.linalg.norm(reference) * np.linalg.norm(converted)))
        print(f"  {size[0]}x{size[1]} cosine={cosine:.6f}")
        if cosine < 0.99:
            raise RuntimeError(f"OpenVINO validation failed for {size}: cosine={cosine:.6f}")
    shutil.rmtree(output / "compiled_cache", ignore_errors=True)
    print(f"Ready: {output}")


if __name__ == "__main__":
    main()
