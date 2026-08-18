from __future__ import annotations

import json
import logging
import re
import time
import threading
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from .domain import SearchHit, VlmVerdict
from .gpu_scheduler import GpuPriority, GpuScheduler, InferenceCancelled
from .interfaces import VlmEngine
from .storage import SQLiteAssetRepository
from .text import classify_query, metadata_filters, normalize_text


LOGGER = logging.getLogger(__name__)


def create_region_review_canvas(image_path: Path, box: Sequence[int] | None) -> Path | None:
    """Create one square full-image + enlarged-region input without textual hints."""
    if not box or len(box) != 4:
        return None
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) >= image.width * image.height * 0.65:
        return None
    crop = image.crop((x1, y1, x2, y2))
    canvas = Image.new("RGB", (896, 896), (18, 24, 36))
    full = ImageOps.contain(image, (584, 864), Image.Resampling.LANCZOS)
    detail = ImageOps.contain(crop, (280, 864), Image.Resampling.LANCZOS)
    canvas.paste(full, ((600 - full.width) // 2, (896 - full.height) // 2))
    canvas.paste(detail, (600 + (296 - detail.width) // 2, (896 - detail.height) // 2))
    handle = tempfile.NamedTemporaryFile(prefix="visual-search-review-", suffix=".jpg", delete=False)
    path = Path(handle.name)
    handle.close()
    canvas.save(path, "JPEG", quality=90)
    return path


def visual_review_target(original_request: str) -> str:
    """Remove deterministic file/date filters while retaining the visual request."""
    target = original_request.strip()
    target = re.sub(r"(?i)(?:folder|directory|資料夾)\s*[:：]\s*[^\s]+", " ", target)
    target = re.sub(r"\b(?:19|20)\d{2}(?:[-/.年](?:0?[1-9]|1[0-2]))?(?:[-/.月](?:0?[1-9]|[12]\d|3[01]))?", " ", target)
    target = re.sub(r"(?i)(?:^|\s|\.)(?:png|jpe?g|webp|bmp|tiff?)(?:$|\s)", " ", target)
    target = " ".join(target.split())
    return target or original_request.strip()


def build_review_prompt(original_request: str, visual_target: str,
                        relevant_ocr: Sequence[str] = ()) -> str:
    ocr = [str(line).strip() for line in relevant_ocr if str(line).strip()][:2]
    total = 0
    clipped: list[str] = []
    for line in ocr:
        remaining = 160 - total
        if remaining <= 0:
            break
        clipped.append(line[:remaining])
        total += len(clipped[-1])
    ocr_block = "none" if not clipped else "\n".join(f"- {line}" for line in clipped)
    return (
        "You are the final visual relevance judge for a private local image search application.\n"
        "Judge the image itself against the user's exact request. The original request is authoritative.\n"
        "Do not invent details. Set hard_fail=true only when an explicit visible constraint is clearly contradicted.\n"
        "Return exactly one compact JSON object and no markdown, analysis, or extra keys.\n"
        'Schema: {"match": integer 0..100, "hard_fail": boolean}\n\n'
        f"Original user request:\n{original_request}\n\n"
        f"Visual review target:\n{visual_target}\n\n"
        f"Relevant OCR (may contain recognition errors):\n{ocr_block}"
    )


def parse_verdict(value: str) -> VlmVerdict:
    stripped = value.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValueError("VLM response is not a single JSON object")
    payload = json.loads(stripped)
    if not isinstance(payload, dict) or set(payload) != {"match", "hard_fail"}:
        raise ValueError("VLM response must contain only match and hard_fail")
    return VlmVerdict(payload["match"], payload["hard_fail"])


class NoopVlmEngine(VlmEngine):
    def review(self, original_request: str, visual_target: str, image: Path,
               relevant_ocr: Sequence[str] = ()) -> VlmVerdict:
        raise RuntimeError("VLM is not installed")

    def runtime_details(self) -> list[dict[str, object]]:
        return [{"name": "Qwen3.5 最終審核", "backend": "未安裝", "execution_devices": [], "loaded": False}]


class QwenOpenVinoVlmEngine(VlmEngine):
    """Qwen3.5/Qwen3-VL with GenAI fast path and Optimum Intel fallback."""

    def __init__(self, model_dir: Path, device: str = "GPU", image_size: int = 448):
        import openvino as ov
        from .engines import select_openvino_device

        self.model_dir = Path(model_dir)
        self.device = select_openvino_device(ov.Core(), device)
        self.image_size = image_size if image_size in {336, 384, 448} else 448
        self.backend = ""
        self.execution_devices: list[str] = []
        self._pipeline = None
        self._model = None
        self._processor = None
        genai_error: Exception | None = None
        try:
            self._load_genai()
        except Exception as exc:
            genai_error = exc
            LOGGER.warning("OpenVINO GenAI VLM unavailable; trying Optimum Intel: %s", exc)
        if self._pipeline is None:
            try:
                self._load_optimum()
            except Exception as optimum_error:
                raise RuntimeError(
                    f"Qwen OpenVINO failed with GenAI ({genai_error}) and Optimum Intel ({optimum_error})"
                ) from optimum_error

    def _load_genai(self) -> None:
        import openvino_genai as ov_genai

        self._pipeline = ov_genai.VLMPipeline(str(self.model_dir), self.device)
        self.backend = "OpenVINO GenAI"
        self.execution_devices = [self.device]

    def _load_optimum(self) -> None:
        from transformers import AutoProcessor
        from optimum.intel.openvino import OVModelForVisualCausalLM

        self._processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self._model = OVModelForVisualCausalLM.from_pretrained(
            self.model_dir, device=self.device, local_files_only=True,
            ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(self.model_dir / "compiled_cache")},
        )
        self.backend = "Optimum Intel OpenVINO"
        self.execution_devices = [self.device]

    def _image(self, path: Path) -> Image.Image:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return image

    def review(self, original_request: str, visual_target: str, image: Path,
               relevant_ocr: Sequence[str] = ()) -> VlmVerdict:
        prompt = build_review_prompt(original_request, visual_target, relevant_ocr)
        prepared = self._image(image)
        if self._pipeline is not None:
            import openvino as ov
            import openvino_genai as ov_genai

            generation = ov_genai.GenerationConfig()
            generation.max_new_tokens = 24
            generation.do_sample = False
            schema = {
                "type": "object",
                "properties": {
                    "match": {"type": "integer", "minimum": 0, "maximum": 100},
                    "hard_fail": {"type": "boolean"},
                },
                "required": ["match", "hard_fail"],
                "additionalProperties": False,
            }
            generation.structured_output_config = ov_genai.StructuredOutputConfig(
                json_schema=json.dumps(schema, separators=(",", ":")),
            )
            tensor = ov.Tensor(np.asarray(prepared, dtype=np.uint8)[None])
            output = self._pipeline.generate(prompt, image=tensor, generation_config=generation)
            texts = getattr(output, "texts", None)
            if not texts:
                raise ValueError("OpenVINO GenAI returned no decoded VLM text")
            return parse_verdict(str(texts[0]))
        if self._model is None or self._processor is None:
            raise RuntimeError("VLM runtime is not loaded")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[prepared], return_tensors="pt")
        output = self._model.generate(**inputs, max_new_tokens=24, do_sample=False)
        prompt_length = int(inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0]
        return parse_verdict(decoded)

    def smoke_test(self, image: Path) -> None:
        verdict = self.review("找藍色圖片", "藍色圖片", image)
        if not 0 <= verdict.match <= 100:
            raise RuntimeError("VLM smoke test returned an invalid score")

    def runtime_details(self) -> list[dict[str, object]]:
        return [{
            "name": "Qwen3.5 9B 最終審核", "backend": self.backend,
            "requested_device": self.device, "execution_devices": self.execution_devices,
            "selected_device": self.device, "loaded": self._pipeline is not None or self._model is not None,
        }]


class LazyVlmEngine(VlmEngine):
    def __init__(self, model_dir: Path, device: str = "GPU", image_size: int = 448,
                 display_name: str = "Qwen3.5 9B"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.image_size = image_size
        self.display_name = display_name
        self._engine: QwenOpenVinoVlmEngine | None = None
        self._lock = threading.Lock()

    def _load(self) -> QwenOpenVinoVlmEngine:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = QwenOpenVinoVlmEngine(self.model_dir, self.device, self.image_size)
        return self._engine

    def review(self, original_request: str, visual_target: str, image: Path,
               relevant_ocr: Sequence[str] = ()) -> VlmVerdict:
        return self._load().review(original_request, visual_target, image, relevant_ocr)

    def runtime_details(self) -> list[dict[str, object]]:
        if self._engine is not None:
            return self._engine.runtime_details()
        return [{
            "name": self.display_name + " 最終審核", "backend": "OpenVINO（等待首次使用）",
            "requested_device": self.device, "execution_devices": [], "selected_device": self.device,
            "loaded": False, "installed": self.model_dir.is_dir(),
        }]


class VlmReranker:
    def __init__(self, repository: SQLiteAssetRepository, engine: VlmEngine,
                 scheduler: GpuScheduler, policy: str = "auto", candidate_count: int = 30,
                 timeout_seconds: int = 60):
        self.repository = repository
        self.engine = engine
        self.scheduler = scheduler
        self.policy = policy if policy in {"auto", "always", "off"} else "auto"
        self.candidate_count = max(1, min(int(candidate_count), 30))
        self.timeout_seconds = max(5, int(timeout_seconds))

    def should_run(self, query: str, hits: Sequence[SearchHit]) -> bool:
        if self.policy == "off" or isinstance(self.engine, NoopVlmEngine) or not hits:
            return False
        if self.policy == "always":
            return True
        return classify_query(query) in {"visual", "mixed"} and not all(hit.protected_exact_match for hit in hits[:10])

    def relevant_ocr(self, query: str, hit: SearchHit) -> list[str]:
        if hit.asset.id is None or classify_query(query) not in {"mixed", "identifier"}:
            return []
        tokens = set(normalize_text(query).split())
        ranked: list[tuple[int, float, str]] = []
        for line in self.repository.ocr_lines(hit.asset.id):
            normalized = set(normalize_text(line.text).split())
            overlap = len(tokens & normalized)
            if overlap:
                ranked.append((overlap, line.confidence, line.text))
        ranked.sort(reverse=True)
        result: list[str] = []
        used = 0
        for _overlap, _confidence, line in ranked:
            if len(result) >= 2 or used >= 160:
                break
            result.append(line[:160 - used])
            used += len(result[-1])
        return result

    def rerank(self, query: str, hits: Sequence[SearchHit], generation: int,
               progress=None) -> list[SearchHit]:
        deterministic = list(hits)
        if not self.should_run(query, deterministic):
            return deterministic[:10]
        candidates = deterministic[:self.candidate_count]
        target = visual_review_target(query)
        started = time.monotonic()
        verdicts: list[tuple[SearchHit, VlmVerdict]] = []
        try:
            for index, hit in enumerate(candidates, 1):
                remaining = self.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("VLM Top 30 review timed out")
                review_image = create_region_review_canvas(Path(hit.asset.path), hit.matched_region)
                try:
                    verdict = self.scheduler.run(
                        lambda current=hit, selected=review_image: self.engine.review(
                            query, target, selected or Path(current.asset.path), self.relevant_ocr(query, current)
                        ),
                        GpuPriority.VLM_REVIEW, generation=generation, timeout=remaining,
                    )
                finally:
                    if review_image is not None:
                        review_image.unlink(missing_ok=True)
                verdicts.append((hit, verdict))
                if progress:
                    progress(index, len(candidates))
            accepted: list[SearchHit] = []
            for hit, verdict in verdicts:
                hard_fail = bool(verdict.hard_fail and not hit.protected_exact_match)
                hit.vlm_score = verdict.match / 100.0
                hit.vlm_hard_fail = hard_fail
                hit.rerank_source = "qwen3.5"
                hit.score = 0.70 * hit.vlm_score + 0.30 * hit.deterministic_score
                if not hard_fail:
                    hit.reasons = list(dict.fromkeys([f"VLM 最終審核：{verdict.match}", *hit.reasons]))
                    accepted.append(hit)
            return sorted(accepted, key=lambda item: item.score, reverse=True)[:10]
        except Exception as exc:
            LOGGER.warning("VLM rerank failed; deterministic results retained: %s", exc)
            for hit in deterministic[:10]:
                hit.score = hit.deterministic_score or hit.score
                hit.vlm_score = None
                hit.vlm_hard_fail = False
                hit.rerank_source = "deterministic-fallback"
            return deterministic[:10]
