from __future__ import annotations

import json
import hashlib
import logging
import re
import time
import threading
import tempfile
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from .domain import SearchHit, SearchMode, SearchProgress, SearchSession, VlmBatchVerdict, VlmVerdict
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


def build_batch_review_prompt(original_request: str, visual_target: str,
                              relevant_ocr: Sequence[Sequence[str]]) -> str:
    evidence: list[str] = []
    for index, lines in enumerate(relevant_ocr):
        clipped = [str(line).strip() for line in lines if str(line).strip()][:2]
        remaining = 160
        kept: list[str] = []
        for line in clipped:
            if remaining <= 0:
                break
            kept.append(line[:remaining])
            remaining -= len(kept[-1])
        evidence.append(f"Image {index} OCR: " + (" | ".join(kept) if kept else "none"))
    return (
        "You are the final visual relevance judge for a private local image search application.\n"
        "Judge every image independently against the user's exact request; image order defines candidate_id.\n"
        "The original request is authoritative. Do not invent details. Set hard_fail=true only when an "
        "explicit visible constraint is clearly contradicted.\n"
        "match is an integer percentage, never a 0/1 boolean: 90-100 clearly matches, 70-89 mostly matches, "
        "40-69 is uncertain/partial, and 0-39 does not match.\n"
        "Return exactly one compact JSON object and no markdown, analysis, or extra keys.\n"
        'Schema example: {"results":[{"candidate_id":0,"match":91,"hard_fail":false}]}\n\n'
        f"Original user request:\n{original_request}\n\n"
        f"Visual review target:\n{visual_target}\n\n"
        + "\n".join(evidence)
    )


def parse_batch_verdicts(value: str, expected: int) -> list[VlmBatchVerdict]:
    stripped = value.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValueError(f"VLM batch response is not one JSON object: {stripped[:300]!r}")
    payload = json.loads(stripped)
    if not isinstance(payload, dict) or set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise ValueError("VLM batch response must contain only a results array")
    if len(payload["results"]) != expected:
        raise ValueError(f"VLM returned {len(payload['results'])} verdicts for {expected} images")
    result: list[VlmBatchVerdict] = []
    for item in payload["results"]:
        if not isinstance(item, dict) or set(item) != {"candidate_id", "match", "hard_fail"}:
            raise ValueError("Each VLM batch verdict must contain candidate_id, match and hard_fail")
        candidate_id = item["candidate_id"]
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            raise ValueError("VLM candidate_id must be an integer")
        result.append(VlmBatchVerdict(candidate_id, VlmVerdict(item["match"], item["hard_fail"])))
    if sorted(item.candidate_id for item in result) != list(range(expected)):
        raise ValueError("VLM candidate ids do not match the supplied image order")
    return sorted(result, key=lambda item: item.candidate_id)


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
        self.model_revision = self._model_revision()
        self.batch_size = 4
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

    def _model_revision(self) -> str:
        for name in (".verified.json", "model-manifest.json", "manifest.json", "config.json"):
            path = self.model_dir / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                revision = payload.get("revision") or payload.get("_commit_hash") or payload.get("model_revision")
                if revision:
                    return str(revision)
            except (OSError, ValueError, TypeError):
                continue
        return self.model_dir.name

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
        return self.review_batch(original_request, visual_target, [image], [relevant_ocr])[0].verdict

    def review_batch(self, original_request: str, visual_target: str,
                     images: Sequence[Path], relevant_ocr: Sequence[Sequence[str]] = ()) -> list[VlmBatchVerdict]:
        if not images:
            return []
        if len(images) > self.batch_size:
            raise ValueError(f"VLM batch supports at most {self.batch_size} images")
        rows = list(relevant_ocr)
        rows.extend([()] * (len(images) - len(rows)))
        prompt = build_batch_review_prompt(original_request, visual_target, rows[:len(images)])
        prepared = [self._image(image) for image in images]
        if self._pipeline is not None:
            import openvino as ov
            import openvino_genai as ov_genai

            generation = ov_genai.GenerationConfig()
            # Structured output may still be pretty-printed by the backend;
            # leave enough room for four complete verdict objects.
            generation.max_new_tokens = 32 + 40 * len(images)
            generation.do_sample = False
            schema = {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array", "minItems": len(images), "maxItems": len(images),
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "integer", "minimum": 0, "maximum": len(images) - 1},
                                "match": {"type": "integer", "minimum": 0, "maximum": 100,
                                          "description": "Relevance percentage: 100 is a perfect visual match"},
                                "hard_fail": {"type": "boolean"},
                            },
                            "required": ["candidate_id", "match", "hard_fail"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["results"],
                "additionalProperties": False,
            }
            generation.structured_output_config = ov_genai.StructuredOutputConfig(
                json_schema=json.dumps(schema, separators=(",", ":")),
            )
            tensors = [ov.Tensor(np.asarray(image, dtype=np.uint8)[None]) for image in prepared]
            output = self._pipeline.generate(prompt, images=tensors, generation_config=generation)
            texts = getattr(output, "texts", None)
            if not texts:
                raise ValueError("OpenVINO GenAI returned no decoded VLM text")
            return parse_batch_verdicts(str(texts[0]), len(images))
        if self._model is None or self._processor is None:
            raise RuntimeError("VLM runtime is not loaded")
        content: list[dict[str, str]] = []
        for index in range(len(images)):
            content.extend((
                {"type": "text", "text": f"Image {index}:"},
                {"type": "image"},
            ))
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=prepared, return_tensors="pt")
        output = self._model.generate(**inputs, max_new_tokens=32 + 40 * len(images), do_sample=False)
        prompt_length = int(inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0]
        return parse_batch_verdicts(decoded, len(images))

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
        self._model_revision = self._discover_revision()
        self._engine: QwenOpenVinoVlmEngine | None = None
        self._lock = threading.Lock()

    def _discover_revision(self) -> str:
        for name in (".verified.json", "model-manifest.json", "manifest.json", "config.json"):
            path = self.model_dir / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                revision = payload.get("revision") or payload.get("_commit_hash") or payload.get("model_revision")
                if revision:
                    return str(revision)
            except (OSError, ValueError, TypeError):
                continue
        return self.model_dir.name

    def _load(self) -> QwenOpenVinoVlmEngine:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = QwenOpenVinoVlmEngine(self.model_dir, self.device, self.image_size)
        return self._engine

    def review(self, original_request: str, visual_target: str, image: Path,
               relevant_ocr: Sequence[str] = ()) -> VlmVerdict:
        return self._load().review(original_request, visual_target, image, relevant_ocr)

    @property
    def model_revision(self) -> str:
        return self._engine.model_revision if self._engine is not None else self._model_revision

    @property
    def batch_size(self) -> int:
        return self._engine.batch_size if self._engine is not None else 4

    def review_batch(self, original_request: str, visual_target: str,
                     images: Sequence[Path], relevant_ocr: Sequence[Sequence[str]] = ()) -> list[VlmBatchVerdict]:
        return self._load().review_batch(original_request, visual_target, images, relevant_ocr)

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
                 timeout_seconds: int = 60, match_threshold: int = 70):
        self.repository = repository
        self.engine = engine
        self.scheduler = scheduler
        self.policy = policy if policy in {"auto", "always", "off"} else "auto"
        # Kept for configuration-file compatibility; complete search has no candidate cap.
        self.candidate_count = 0
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.match_threshold = int(match_threshold) if int(match_threshold) in {60, 70, 85} else 70

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

    @staticmethod
    def _query_hash(query: str) -> str:
        canonical = " ".join(unicodedata.normalize("NFKC", query).split()).casefold()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _region_key(hit: SearchHit) -> str:
        return json.dumps(hit.matched_region or [], separators=(",", ":"))

    def _model_revision(self) -> str:
        return str(getattr(self.engine, "model_revision", self.engine.__class__.__name__))

    def _image_size(self) -> int:
        return int(getattr(self.engine, "image_size", 448))

    @staticmethod
    def _emit(progress, session: SearchSession) -> None:
        if progress is None:
            return
        snapshot = SearchProgress(
            candidate_total=session.candidate_total,
            reviewed=session.reviewed,
            qualified=session.qualified,
            rejected=session.rejected,
            cached=session.cached,
            remaining=max(0, session.candidate_total - session.reviewed),
            complete=session.complete,
            mode=session.mode,
            results=list(session.results),
            error=session.error,
        )
        try:
            progress(snapshot)
        except TypeError:
            progress(snapshot.reviewed, snapshot.candidate_total)

    def _apply(self, hit: SearchHit, verdict: VlmVerdict, source: str) -> bool:
        hard_fail = bool(verdict.hard_fail and not hit.protected_exact_match)
        qualified = hit.protected_exact_match or (not hard_fail and verdict.match >= self.match_threshold)
        hit.vlm_score = verdict.match / 100.0
        hit.vlm_hard_fail = hard_fail
        hit.rerank_source = source
        hit.score = 0.70 * hit.vlm_score + 0.30 * hit.deterministic_score
        if qualified:
            hit.reasons = list(dict.fromkeys([f"VLM 最終審核：{verdict.match}", *hit.reasons]))
        return qualified

    def _review_batch(self, query: str, target: str, hits: Sequence[SearchHit], generation: int) -> list[VlmVerdict]:
        review_images: list[Path] = []
        temporary: list[Path] = []
        try:
            for hit in hits:
                review_image = create_region_review_canvas(Path(hit.asset.path), hit.matched_region)
                if review_image is not None:
                    temporary.append(review_image)
                review_images.append(review_image or Path(hit.asset.path))
            evidence = [self.relevant_ocr(query, hit) for hit in hits]
            supports_batch = callable(getattr(self.engine, "review_batch", None))
            for attempt in range(2 if supports_batch else 0):
                try:
                    rows = self.scheduler.run(
                        lambda: self.engine.review_batch(query, target, review_images, evidence),
                        GpuPriority.VLM_REVIEW, generation=generation, timeout=self.timeout_seconds,
                    )
                    if len(rows) != len(hits):
                        raise ValueError("VLM batch verdict count mismatch")
                    return [row.verdict for row in sorted(rows, key=lambda item: item.candidate_id)]
                except InferenceCancelled:
                    raise
                except Exception:
                    if attempt:
                        LOGGER.warning("VLM multi-image batch failed twice; using single-image fallback", exc_info=True)
                    else:
                        LOGGER.warning("VLM multi-image batch failed; retrying once", exc_info=True)
            verdicts: list[VlmVerdict] = []
            for hit, image, lines in zip(hits, review_images, evidence):
                verdicts.append(self.scheduler.run(
                    lambda selected=image, current=hit, ocr=lines: self.engine.review(
                        query, target, selected or Path(current.asset.path), ocr,
                    ),
                    GpuPriority.VLM_REVIEW, generation=generation, timeout=self.timeout_seconds,
                ))
            return verdicts
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)

    def rerank_session(self, query: str, hits: Sequence[SearchHit], generation: int,
                       progress=None, mode: SearchMode = SearchMode.SMART_COMPLETE) -> SearchSession:
        deterministic = list(hits)
        session = SearchSession(generation, query, mode, len(deterministic))
        if not self.should_run(query, deterministic):
            for hit in deterministic:
                hit.rerank_source = "deterministic"
            session.results = deterministic
            session.reviewed = len(deterministic)
            session.qualified = len(deterministic)
            session.complete = True
            session.reranker_source = "deterministic"
            self._emit(progress, session)
            return session
        candidates = deterministic
        target = visual_review_target(query)
        query_hash = self._query_hash(query)
        model_revision = self._model_revision()
        image_size = self._image_size()
        accepted: list[SearchHit] = []
        pending: list[SearchHit] = []
        self.repository.purge_vlm_review_cache()
        for hit in candidates:
            if hit.asset.id is None:
                pending.append(hit)
                continue
            verdict = self.repository.cached_vlm_verdict(
                query_hash, hit.asset.id, hit.asset.sha256, self._region_key(hit),
                model_revision, image_size,
            )
            if verdict is None:
                pending.append(hit)
                continue
            session.cached += 1
            session.reviewed += 1
            if self._apply(hit, verdict, "qwen3.5-cache"):
                accepted.append(hit)
                session.qualified += 1
            else:
                session.rejected += 1
        session.results = sorted(accepted, key=lambda item: item.score, reverse=True)
        self._emit(progress, session)
        batch_size = max(1, min(4, int(getattr(self.engine, "batch_size", 4))))
        try:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                verdicts = self._review_batch(query, target, batch, generation)
                cache_rows: list[tuple[int, str, str, VlmVerdict]] = []
                for hit, verdict in zip(batch, verdicts):
                    if hit.asset.id is not None and self.repository.get(hit.asset.id) is not None:
                        cache_rows.append((hit.asset.id, hit.asset.sha256, self._region_key(hit), verdict))
                    session.reviewed += 1
                    if self._apply(hit, verdict, "qwen3.5"):
                        accepted.append(hit)
                        session.qualified += 1
                    else:
                        session.rejected += 1
                self.repository.save_vlm_verdicts(query_hash, model_revision, image_size, cache_rows)
                session.results = sorted(accepted, key=lambda item: item.score, reverse=True)
                self._emit(progress, session)
                if mode == SearchMode.EXHAUSTIVE and ((offset // batch_size) + 1) % 4 == 0:
                    self.scheduler.run(lambda: None, GpuPriority.BACKGROUND_INDEX, timeout=self.timeout_seconds)
            session.complete = True
            session.reranker_source = "qwen3.5"
            session.results = sorted(accepted, key=lambda item: item.score, reverse=True)
            self._emit(progress, session)
            return session
        except Exception as exc:
            LOGGER.warning("VLM complete review paused; verified results retained: %s", exc)
            session.error = str(exc)
            session.complete = False
            session.reranker_source = "qwen3.5-incomplete"
            session.results = sorted(accepted, key=lambda item: item.score, reverse=True)
            self._emit(progress, session)
            return session

    def rerank(self, query: str, hits: Sequence[SearchHit], generation: int,
               progress=None, mode: SearchMode = SearchMode.SMART_COMPLETE) -> list[SearchHit]:
        return self.rerank_session(query, hits, generation, progress, mode).results
