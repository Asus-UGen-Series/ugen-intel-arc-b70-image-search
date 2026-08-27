from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .domain import OcrLine, Vector
from .interfaces import EmbeddingEngine, OcrEngine

LOGGER = logging.getLogger(__name__)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class PaddleOcrEngine(OcrEngine):
    def __init__(self, minimum_confidence: float = 0.55):
        self.minimum_confidence = minimum_confidence
        self._engine = None

    def _load(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            # Paddle 3.3's oneDNN executor currently rejects an attribute used by
            # PP-OCRv6 on some Windows CPUs. Disabling MKL-DNN is slower but makes
            # the packaged CPU fallback reliable; the release OpenVINO backend
            # replaces this development adapter.
            self._engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        return self._engine

    def extract(self, image: Path) -> list[OcrLine]:
        result = self._load().predict(str(image))
        lines: list[OcrLine] = []
        for page in result:
            payload = getattr(page, "json", page)
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload = payload.get("res", payload) if isinstance(payload, dict) else {}
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            boxes = payload.get("rec_polys", payload.get("dt_polys", []))
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 0.0
                if text.strip() and score >= self.minimum_confidence:
                    box = np.asarray(boxes[index]).tolist() if index < len(boxes) else []
                    lines.append(OcrLine(text.strip(), score, box))
        return lines

    def runtime_details(self) -> list[dict[str, object]]:
        return [{
            "name": "PP-OCRv6 偵測與辨識（相容模式）",
            "backend": "PaddlePaddle",
            "requested_device": "CPU",
            "execution_devices": ["CPU"],
            "selected_device": "CPU",
            "loaded": self._engine is not None,
        }]


class NoopOcrEngine(OcrEngine):
    def extract(self, image: Path) -> list[OcrLine]:
        return []

    def runtime_details(self) -> list[dict[str, object]]:
        return [{"name": "OCR", "backend": "停用", "execution_devices": [], "loaded": False}]


class OpenVinoSiglipEngine(EmbeddingEngine):
    """Runs the statically-shaped OpenVINO SigLIP 2 NaFlex package.

    Hugging Face computes NaFlex positional embeddings with Python control flow,
    which would freeze one aspect ratio during tracing. The release exporter
    therefore makes resized positional embeddings an explicit model input. The
    small interpolation below stays on CPU while the expensive transformer runs
    with a fixed 256-patch shape on the Intel GPU.
    """
    model_version = "siglip2-so400m-naflex-256-fp16-v2"

    def __init__(self, model_dir: Path, device: str = "AUTO"):
        import openvino as ov
        # Transformers only supplies the processor/tokenizer in the shipped
        # OpenVINO application. PyTorch is deliberately absent, so hide its
        # generic backend availability notice from end users.
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoProcessor

        self.model_dir = Path(model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, backend="pil")
        core = ov.Core()
        preferred = select_openvino_device(core, device)
        self.device = preferred
        cache = self.model_dir / "compiled_cache"
        cache.mkdir(exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache)})
        config = serial_inference_config(preferred)
        self.vision = core.compile_model(model_dir / "vision_model.xml", preferred, config)
        self.text_model = core.compile_model(model_dir / "text_model.xml", preferred, config)
        self.vision_request = self.vision.create_infer_request()
        self.text_request = self.text_model.create_infer_request()
        self.vision_execution_devices = compiled_execution_devices(self.vision, preferred)
        self.text_execution_devices = compiled_execution_devices(self.text_model, preferred)
        self._base_positions = np.load(model_dir / "position_embedding.npy").astype(np.float32)
        if self._base_positions.ndim != 3:
            raise ValueError("position_embedding.npy must have shape [height, width, dimensions]")
        self._position_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._dimensions = int(self.vision.output(0).partial_shape[-1].get_length())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def image(self, image: Path) -> Vector:
        with Image.open(image) as source:
            output = self._image_from_pil(source.convert("RGB"))
        return normalize_vector(output)

    def image_regions(self, image: Path, boxes) -> list[Vector]:
        with Image.open(image) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            crops = [oriented.crop(tuple(int(value) for value in box)) for box in boxes]
        if not crops:
            return []
        prepared = [self.processor(images=[crop], return_tensors="np") for crop in crops]
        pixel_values = np.concatenate(
            [np.asarray(item["pixel_values"], dtype=np.float32) for item in prepared], axis=0
        )
        pixel_masks = np.concatenate(
            [np.asarray(item["pixel_attention_mask"], dtype=np.int64) for item in prepared], axis=0
        )
        positions = []
        for item, values in zip(prepared, pixel_values):
            height, width = np.asarray(item["spatial_shapes"])[0].tolist()
            cache_key = (int(height), int(width), int(values.shape[0]))
            position = self._position_cache.get(cache_key)
            if position is None:
                position = resize_position_embeddings(self._base_positions, height, width, values.shape[0])
                self._position_cache[cache_key] = position
            positions.append(position)
        output = infer_named(self.vision, {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_masks,
            "position_embeddings": np.concatenate(positions, axis=0),
        }, self.vision_request)
        return [normalize_vector(row) for row in np.asarray(output)]

    def _image_from_pil(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="np")
        pixel_values = np.asarray(inputs["pixel_values"], dtype=np.float32)
        pixel_mask = np.asarray(inputs["pixel_attention_mask"], dtype=np.int64)
        height, width = np.asarray(inputs["spatial_shapes"])[0].tolist()
        cache_key = (int(height), int(width), int(pixel_values.shape[1]))
        positions = self._position_cache.get(cache_key)
        if positions is None:
            positions = resize_position_embeddings(self._base_positions, height, width, pixel_values.shape[1])
            self._position_cache[cache_key] = positions
        values = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_mask,
            "position_embeddings": positions,
        }
        output = infer_named(self.vision, values, self.vision_request)
        return normalize_vector(output)

    def text(self, text: str) -> Vector:
        inputs = self.processor(text=[text], padding="max_length", return_tensors="np")
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = np.ones_like(inputs["input_ids"], dtype=np.int64)
        values = {name: inputs[name] for name in ("input_ids", "attention_mask") if name in inputs}
        output = infer_named(self.text_model, values, self.text_request)
        return normalize_vector(output)

    def smoke_test(self) -> None:
        image_vector = self._image_from_pil(Image.new("RGB", (960, 540), (80, 120, 180)))
        text_vector = self.text("一張藍色的橫向照片")
        for name, vector in (("image", image_vector), ("text", text_vector)):
            if vector.size != self.dimensions or not np.isfinite(vector).all() or not np.linalg.norm(vector):
                raise RuntimeError(f"OpenVINO {name} smoke test failed")

    def runtime_details(self) -> list[dict[str, object]]:
        return [
            {
                "name": "SigLIP 2 圖片編碼",
                "backend": "OpenVINO FP16",
                "requested_device": self.device,
                "execution_devices": self.vision_execution_devices,
                "selected_device": self.device,
                "performance_hint": "LATENCY",
                "optimal_infer_requests": safe_compiled_property(
                    self.vision, "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
                ),
                "loaded": True,
            },
            {
                "name": "SigLIP 2 文字編碼",
                "backend": "OpenVINO FP16",
                "requested_device": self.device,
                "execution_devices": self.text_execution_devices,
                "selected_device": self.device,
                "performance_hint": "LATENCY",
                "optimal_infer_requests": safe_compiled_property(
                    self.text_model, "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
                ),
                "loaded": True,
            },
        ]


def infer_named(compiled_model, values: dict[str, np.ndarray], infer_request=None) -> np.ndarray:
    """Run an IR while tolerating harmless port-name suffixes from conversion."""
    prepared: dict[object, np.ndarray] = {}
    remaining = list(values.items())
    for port in compiled_model.inputs:
        names = set(port.names)
        match = next(((name, value) for name, value in remaining if name in names), None)
        if match is None:
            if not remaining:
                raise KeyError(f"No value for OpenVINO input {port.any_name}")
            match = remaining[0]
        prepared[port] = match[1]
        remaining.remove(match)
    request = infer_request or compiled_model.create_infer_request()
    request.infer(prepared)
    return np.array(request.get_output_tensor(0).data, copy=True)


def serial_inference_config(device: str) -> dict[str, str]:
    """The application serializes GPU work, so extra OpenVINO streams only waste memory."""
    config = {"PERFORMANCE_HINT": "LATENCY"}
    if str(device).upper().startswith("GPU"):
        config["NUM_STREAMS"] = "1"
    return config


def safe_compiled_property(compiled_model, name: str):
    try:
        return compiled_model.get_property(name)
    except Exception:
        return None


def resize_position_embeddings(base: np.ndarray, height: int, width: int, max_length: int = 256) -> np.ndarray:
    """Resize an HWC position grid like PyTorch bilinear + antialias.

    This deliberately returns a fixed [1, max_length, dimensions] tensor. Padded
    rows repeat the first position exactly like Transformers' SigLIP2 code; the
    pixel mask prevents them from participating in attention.
    """
    height, width, max_length = int(height), int(width), int(max_length)
    if height <= 0 or width <= 0 or height * width > max_length:
        raise ValueError("Invalid NaFlex patch grid")
    source_h, source_w, dimensions = base.shape
    y_weights = _antialias_linear_weights(source_h, height)
    x_weights = _antialias_linear_weights(source_w, width)
    # Separable resize keeps the runtime dependency-free and matches the
    # antialiased NaFlex interpolation used by Transformers closely.
    resized_grid = np.einsum("yh,hwc,xw->yxc", y_weights, base, x_weights, optimize=True)
    resized = resized_grid.reshape(height * width, dimensions)
    padded = np.empty((max_length, dimensions), dtype=np.float32)
    padded[: height * width] = resized
    padded[height * width :] = resized[0]
    return padded[None, ...]


def _antialias_linear_weights(source: int, target: int) -> np.ndarray:
    """Build align_corners=False triangle-filter weights for one dimension."""
    scale = source / target
    support = max(scale, 1.0)
    weights = np.zeros((target, source), dtype=np.float32)
    for destination in range(target):
        center = (destination + 0.5) * scale - 0.5
        first = max(0, int(np.floor(center - support + 1.0)))
        last = min(source - 1, int(np.ceil(center + support - 1.0)))
        indexes = np.arange(first, last + 1)
        values = np.maximum(0.0, 1.0 - np.abs(indexes - center) / support)
        values /= values.sum()
        weights[destination, indexes] = values
    return weights


class ColorFallbackEmbedding(EmbeddingEngine):
    """Small deterministic fallback so the app remains usable before models arrive."""
    model_version = "color-fallback-v1"

    @property
    def dimensions(self) -> int:
        return 48

    def image(self, image: Path) -> Vector:
        with Image.open(image) as source:
            rgb = np.asarray(source.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        hist = np.concatenate([np.histogram(rgb[..., channel], bins=16, range=(0, 1), density=True)[0] for channel in range(3)])
        return normalize_vector(hist)

    def text(self, text: str) -> Vector:
        # No fake semantic claim: an empty vector yields OCR/filename-only search.
        return np.zeros(self.dimensions, dtype=np.float32)

    def runtime_details(self) -> list[dict[str, object]]:
        return [{"name": "圖片語意搜尋", "backend": "停用", "execution_devices": [], "loaded": False}]


def compiled_execution_devices(compiled_model, fallback: str) -> list[str]:
    try:
        return [str(item) for item in compiled_model.get_property("EXECUTION_DEVICES")]
    except Exception:
        return [fallback]


def detect_openvino_devices() -> dict[str, object]:
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        selected = select_openvino_device(core)
        names = {}
        for item in devices:
            try:
                names[item] = core.get_property(item, "FULL_DEVICE_NAME")
            except Exception:
                names[item] = item
        return {"available": devices, "device_names": names, "selected": selected, "ok": True}
    except Exception as exc:
        return {"available": [], "selected": "CPU", "ok": False, "error": str(exc)}


def select_openvino_device(core, requested: str = "AUTO") -> str:
    available = list(core.available_devices)
    if requested not in ("AUTO", "GPU") and requested in available:
        return requested
    gpus = [item for item in available if item.startswith("GPU")]
    for item in gpus:
        try:
            if "intel" in str(core.get_property(item, "FULL_DEVICE_NAME")).lower():
                return item
        except Exception:
            pass
    # This product is for Intel Arc. Do not silently run the workload on a
    # non-Intel GPU and present it as B70 acceleration.
    return "CPU"
