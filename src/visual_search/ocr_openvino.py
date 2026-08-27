from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .domain import OcrLine
from .interfaces import OcrEngine
from .text import code_search_terms

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectedTextBox:
    box: np.ndarray
    confidence: float
    source_pass: str = "full"


class OpenVinoPaddleOcrEngine(OcrEngine):
    """PP-OCRv6 medium detection + recognition on OpenVINO.

    Neural inference is compiled explicitly for the Intel GPU. Image decode,
    geometric box processing, crops and CTC decoding remain on CPU by design.
    """

    DET_BUCKETS = ((320, 960), (640, 960), (960, 960), (960, 640), (960, 320))
    REC_WIDTH_BUCKETS = (160, 320, 640, 1280, 3200)
    model_version = "pp-ocrv6-medium-openvino-fp16-v3"

    def __init__(
        self,
        model_dir: Path,
        device: str = "GPU",
        minimum_confidence: float = 0.55,
        recognition_batch_size: int = 8,
    ):
        import openvino as ov
        import yaml

        from .engines import select_openvino_device, serial_inference_config

        self.model_dir = Path(model_dir)
        self.minimum_confidence = minimum_confidence
        self.recognition_batch_size = max(1, min(int(recognition_batch_size), 32))
        self.last_diagnostics: dict[str, object] = {}
        core = ov.Core()
        self.requested_device = device
        self.device = select_openvino_device(core, device)
        cache = self.model_dir / "compiled_cache"
        cache.mkdir(exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache)})
        config = serial_inference_config(self.device)
        self.det_model = core.compile_model(self.model_dir / "det_model.xml", self.device, config)
        self.rec_model = core.compile_model(self.model_dir / "rec_model.xml", self.device, config)
        self.det_request = self.det_model.create_infer_request()
        self.rec_request = self.rec_model.create_infer_request()
        self.det_execution_devices = execution_devices(self.det_model, self.device)
        self.rec_execution_devices = execution_devices(self.rec_model, self.device)
        self.det_optimal_requests = integer_property(self.det_model, "OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        self.rec_optimal_requests = integer_property(self.rec_model, "OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        config_data = yaml.safe_load((self.model_dir / "rec_inference.yml").read_text(encoding="utf-8"))
        self.characters = ["blank"] + list(config_data["PostProcess"]["character_dict"])

    def extract(self, image: Path) -> list[OcrLine]:
        source = decode_image(image)
        return self.extract_array(source, image.suffix.lower())

    def extract_array(self, source: np.ndarray, image_suffix: str = "") -> list[OcrLine]:
        full = [as_detection(item, "full") for item in self.detect(source)]
        full_scale = float(getattr(self, "_last_detection_scale", 1.0))
        detections = list(full)
        tile_count = 0
        skipped_tiles = 0
        if should_use_tiles(source, full_scale, len(full), image_suffix):
            height, width = source.shape[:2]
            for x, y, tile_width, tile_height in tile_regions(width, height):
                tile = source[y : y + tile_height, x : x + tile_width]
                if not tile_has_text_signal(tile):
                    skipped_tiles += 1
                    continue
                tile_count += 1
                tile_detections = self.detect(
                    tile, threshold=0.18, box_threshold=0.35,
                    source_pass=f"tile:{x},{y}", offset=(x, y),
                )
                detections.extend(as_detection(item, f"tile:{x},{y}") for item in tile_detections)

        before_dedup = len(detections)
        detections = deduplicate_detections(detections)
        self.last_diagnostics = {
            "full_scale": round(full_scale, 4),
            "full_boxes": len(full),
            "tile_count": tile_count,
            "skipped_tiles": skipped_tiles,
            "boxes_before_dedup": before_dedup,
            "boxes_after_dedup": len(detections),
            "tiling_enabled": tile_count > 0,
        }
        if not detections:
            return []
        crops = [rotated_crop(source, item.box) for item in detections]
        recognized = self.recognize(crops)
        # Preserve every non-empty recognition result for diagnostics. The
        # searchable flag controls FTS inclusion without destroying raw OCR.
        lines = [
            OcrLine(
                text=text,
                confidence=score,
                box=np.asarray(item.box, dtype=np.float32).tolist(),
                searchable=score >= self.minimum_confidence,
                detection_confidence=item.confidence,
                source_pass=item.source_pass,
                search_terms=code_search_terms(text),
            )
            for item, (text, score) in zip(detections, recognized, strict=True)
            if text.strip()
        ]
        return lines

    def detect(
        self,
        source: np.ndarray,
        threshold: float = 0.2,
        box_threshold: float = 0.45,
        source_pass: str = "full",
        offset: tuple[int, int] = (0, 0),
    ) -> list[DetectedTextBox]:
        tensor, scale, original_size = preprocess_detection(source, self.DET_BUCKETS)
        self._last_detection_scale = scale
        result = infer_first(self.det_model, tensor, self.det_request)
        probability = np.asarray(result).squeeze()
        if probability.ndim != 2:
            raise RuntimeError(f"Unexpected OCR detection output: {np.asarray(result).shape}")
        boxes = boxes_from_probability(
            probability,
            original_size=original_size,
            input_size=(tensor.shape[2], tensor.shape[3]),
            scale=scale,
            threshold=threshold,
            box_threshold=box_threshold,
            unclip_ratio=1.4,
            max_candidates=3000,
        )
        x_offset, y_offset = offset
        detections = []
        for box, confidence in boxes:
            box = box.copy()
            box[:, 0] += x_offset
            box[:, 1] += y_offset
            detections.append(DetectedTextBox(box, confidence, source_pass))
        return sort_detections(detections)

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        buckets: dict[int, list[int]] = {}
        for index, crop in enumerate(crops):
            buckets.setdefault(recognition_width(crop, self.REC_WIDTH_BUCKETS), []).append(index)
        # Keep a fixed padded batch shape for each width bucket. This prevents
        # the GPU plugin from retaining a new dynamic-shape allocation for every
        # possible remainder batch while indexing thousands of images.
        for target_width, ordered in sorted(buckets.items()):
            batch_size = recognition_batch_capacity(target_width, self.recognition_batch_size)
            for start in range(0, len(ordered), batch_size):
                indices = ordered[start : start + batch_size]
                batch_crops = [crops[index] for index in indices]
                padded = [*batch_crops, *([batch_crops[-1]] * (batch_size - len(batch_crops)))]
                tensor = preprocess_recognition(
                    padded, self.REC_WIDTH_BUCKETS, target_width=target_width,
                )
                logits = infer_first(self.rec_model, tensor, self.rec_request)
                decoded = decode_ctc(np.asarray(logits), self.characters)
                for index, value in zip(indices, decoded[:len(indices)], strict=True):
                    results[index] = value
        return results

    def smoke_test(self) -> None:
        import cv2

        canvas = np.full((256, 768, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "Visual Search 123", (35, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 4)
        det_input, _, _ = preprocess_detection(canvas, self.DET_BUCKETS)
        det_output = infer_first(self.det_model, det_input, self.det_request)
        crop = canvas[70:175, 20:730]
        target_width = recognition_width(crop, self.REC_WIDTH_BUCKETS)
        batch_size = recognition_batch_capacity(target_width, self.recognition_batch_size)
        rec_input = preprocess_recognition(
            [crop] * batch_size, self.REC_WIDTH_BUCKETS, target_width=target_width,
        )
        rec_output = infer_first(self.rec_model, rec_input, self.rec_request)
        if not np.isfinite(det_output).all() or not np.isfinite(rec_output).all():
            raise RuntimeError("OpenVINO OCR smoke test returned invalid values")
        if np.asarray(rec_output).ndim != 3:
            raise RuntimeError(f"Unexpected OCR recognition output: {np.asarray(rec_output).shape}")

    def runtime_details(self) -> list[dict[str, object]]:
        return [
            {
                "name": "PP-OCRv6 文字偵測",
                "backend": "OpenVINO FP16",
                "requested_device": self.requested_device,
                "execution_devices": self.det_execution_devices,
                "selected_device": self.device,
                "optimal_infer_requests": self.det_optimal_requests,
                "performance_hint": "LATENCY",
                "loaded": True,
            },
            {
                "name": "PP-OCRv6 文字辨識",
                "backend": "OpenVINO FP16",
                "requested_device": self.requested_device,
                "execution_devices": self.rec_execution_devices,
                "selected_device": self.device,
                "optimal_infer_requests": self.rec_optimal_requests,
                "performance_hint": "LATENCY",
                "loaded": True,
            },
        ]


def infer_first(compiled_model, tensor: np.ndarray, infer_request=None) -> np.ndarray:
    request = infer_request or compiled_model.create_infer_request()
    request.infer({compiled_model.input(0): tensor})
    return np.array(request.get_output_tensor(0).data, copy=True)


def decode_image(path: Path) -> np.ndarray:
    """Decode an image without passing a Unicode Windows path to cv2.imread."""
    import cv2

    try:
        encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Unable to read image: {path}") from exc
    source = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Unable to decode image: {path}")
    return source


def execution_devices(compiled_model, fallback: str) -> list[str]:
    try:
        value = compiled_model.get_property("EXECUTION_DEVICES")
        return [str(item) for item in value]
    except Exception:
        return [fallback]


def integer_property(compiled_model, name: str) -> int | None:
    try:
        return int(compiled_model.get_property(name))
    except Exception:
        return None


def preprocess_detection(source: np.ndarray, buckets: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, float, tuple[int, int]]:
    import cv2

    original_h, original_w = source.shape[:2]
    ratio = original_w / max(original_h, 1)
    target_h, target_w = min(buckets, key=lambda size: abs(math.log((size[1] / size[0]) / ratio)))
    scale = min(target_w / original_w, target_h / original_h)
    resized_w = max(1, min(target_w, round(original_w * scale)))
    resized_h = max(1, min(target_h, round(original_h * scale)))
    resized = cv2.resize(source, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    # Padding with the ImageNet mean becomes approximately zero after normalization.
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    canvas = np.empty((target_h, target_w, 3), dtype=np.float32)
    canvas[:] = mean * 255.0
    canvas[:resized_h, :resized_w] = resized.astype(np.float32)
    canvas = (canvas / 255.0 - mean) / std
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return tensor, scale, (original_h, original_w)


def recognition_width(crop: np.ndarray, width_buckets: tuple[int, ...]) -> int:
    target_h = 48
    required = math.ceil(target_h * crop.shape[1] / max(crop.shape[0], 1))
    return next((width for width in width_buckets if width >= required), width_buckets[-1])


def recognition_batch_capacity(target_width: int, configured_batch_size: int) -> int:
    # 8 x 320 is the baseline activation budget. Very wide text therefore runs
    # with a smaller batch rather than multiplying its intermediate tensors.
    return max(1, min(int(configured_batch_size), 2560 // max(1, int(target_width))))


def preprocess_recognition(
    crops: list[np.ndarray], width_buckets: tuple[int, ...], target_width: int | None = None,
) -> np.ndarray:
    import cv2

    target_h = 48
    target_w = int(target_width) if target_width is not None else max(
        recognition_width(crop, width_buckets) for crop in crops
    )
    batch = np.zeros((len(crops), 3, target_h, target_w), dtype=np.float32)
    for index, crop in enumerate(crops):
        ratio = crop.shape[1] / max(crop.shape[0], 1)
        resized_w = max(1, min(target_w, math.ceil(target_h * ratio)))
        resized = cv2.resize(crop, (resized_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        resized = (resized / 255.0 - 0.5) / 0.5
        batch[index, :, :, :resized_w] = resized.transpose(2, 0, 1)
    return batch


def boxes_from_probability(
    probability: np.ndarray,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
    scale: float,
    threshold: float,
    box_threshold: float,
    unclip_ratio: float,
    max_candidates: int,
) -> list[tuple[np.ndarray, float]]:
    import cv2

    bitmap = (probability > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    map_h, map_w = probability.shape
    input_h, input_w = input_size
    original_h, original_w = original_size
    boxes: list[tuple[np.ndarray, float]] = []
    for contour in contours[:max_candidates]:
        box, short_side = mini_box(contour)
        if short_side < 3:
            continue
        score = box_score(probability, box)
        if score < box_threshold:
            continue
        expanded = unclip(box, unclip_ratio)
        if expanded is None:
            continue
        box, short_side = mini_box(expanded.reshape(-1, 1, 2).astype(np.float32))
        if short_side < 5:
            continue
        box[:, 0] = np.clip(box[:, 0] / map_w * input_w / scale, 0, original_w - 1)
        box[:, 1] = np.clip(box[:, 1] / map_h * input_h / scale, 0, original_h - 1)
        boxes.append((box.astype(np.float32), float(score)))
    return boxes


def as_detection(value: DetectedTextBox | np.ndarray, source_pass: str = "full") -> DetectedTextBox:
    if isinstance(value, DetectedTextBox):
        return value
    return DetectedTextBox(np.asarray(value, dtype=np.float32), 0.0, source_pass)


def should_use_tiles(source: np.ndarray, scale: float, full_box_count: int, image_suffix: str = "") -> bool:
    height, width = source.shape[:2]
    if scale >= 0.65:
        return False
    screenshot = image_suffix.lower() in {".png", ".bmp"} and max(width, height) > 1600
    return full_box_count >= 4 or screenshot or width / max(height, 1) > 2.2


def axis_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, max(length - tile_size + 1, 1), stride))
    edge = length - tile_size
    if not positions or positions[-1] != edge:
        positions.append(edge)
    return list(dict.fromkeys(positions))


def tile_regions(width: int, height: int, tile_width: int = 1280, tile_height: int = 960,
                 overlap: int = 128) -> list[tuple[int, int, int, int]]:
    xs = axis_positions(width, min(tile_width, width), min(overlap, max(0, min(tile_width, width) - 1)))
    ys = axis_positions(height, min(tile_height, height), min(overlap, max(0, min(tile_height, height) - 1)))
    return [(x, y, min(tile_width, width - x), min(tile_height, height - y)) for y in ys for x in xs]


def tile_has_text_signal(tile: np.ndarray) -> bool:
    import cv2

    if tile.size == 0:
        return False
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
    contrast = float(gray.std())
    if contrast >= 10.0:
        return True
    edges = cv2.Canny(gray, 60, 160)
    return float(np.count_nonzero(edges)) / max(edges.size, 1) >= 0.005


def polygon_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    import cv2

    left = order_points(left).astype(np.float32)
    right = order_points(right).astype(np.float32)
    left_area = abs(float(cv2.contourArea(left)))
    right_area = abs(float(cv2.contourArea(right)))
    if left_area <= 0 or right_area <= 0:
        return 0.0, 0.0
    intersection, _ = cv2.intersectConvexConvex(left, right)
    intersection = max(0.0, float(intersection))
    union = left_area + right_area - intersection
    return intersection / max(union, 1e-6), intersection / min(left_area, right_area)


def deduplicate_detections(detections: list[DetectedTextBox]) -> list[DetectedTextBox]:
    # Prefer high-resolution tile boxes, then stronger detector confidence.
    ordered = sorted(
        detections,
        key=lambda item: (item.source_pass.startswith("tile:"), item.confidence),
        reverse=True,
    )
    accepted: list[DetectedTextBox] = []
    for candidate in ordered:
        if any((lambda overlap: overlap[0] >= 0.5 or overlap[1] >= 0.75)(polygon_overlap(candidate.box, item.box))
               for item in accepted):
            continue
        accepted.append(candidate)
    return sort_detections(accepted)


def sort_detections(detections: list[DetectedTextBox]) -> list[DetectedTextBox]:
    box_lookup = {id(item.box): item for item in detections}
    return [box_lookup[id(box)] for box in sort_boxes([item.box for item in detections])]


def mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    import cv2

    rectangle = cv2.minAreaRect(contour)
    points = cv2.boxPoints(rectangle)
    points = order_points(points)
    return points, min(rectangle[1])


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def box_score(probability: np.ndarray, box: np.ndarray) -> float:
    import cv2

    h, w = probability.shape
    xmin = max(0, int(np.floor(box[:, 0].min())))
    xmax = min(w - 1, int(np.ceil(box[:, 0].max())))
    ymin = max(0, int(np.floor(box[:, 1].min())))
    ymax = min(h - 1, int(np.ceil(box[:, 1].max())))
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = box.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
    return float(cv2.mean(probability[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def unclip(box: np.ndarray, ratio: float) -> np.ndarray | None:
    import pyclipper
    from shapely.geometry import Polygon

    polygon = Polygon(box)
    if polygon.length <= 0 or polygon.area <= 0:
        return None
    distance = polygon.area * ratio / polygon.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    return np.asarray(max(expanded, key=len), dtype=np.float32)


def sort_boxes(boxes: list[np.ndarray]) -> list[np.ndarray]:
    ordered = sorted(boxes, key=lambda box: (float(box[0, 1]), float(box[0, 0])))
    for index in range(1, len(ordered)):
        current = index
        while current > 0 and abs(ordered[current][0, 1] - ordered[current - 1][0, 1]) < 10 and ordered[current][0, 0] < ordered[current - 1][0, 0]:
            ordered[current], ordered[current - 1] = ordered[current - 1], ordered[current]
            current -= 1
    return ordered


def rotated_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    import cv2

    box = order_points(box)
    width = max(int(np.linalg.norm(box[0] - box[1])), int(np.linalg.norm(box[2] - box[3])), 1)
    height = max(int(np.linalg.norm(box[0] - box[3])), int(np.linalg.norm(box[1] - box[2])), 1)
    destination = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(box, destination)
    crop = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


def decode_ctc(logits: np.ndarray, characters: list[str]) -> list[tuple[str, float]]:
    if logits.ndim != 3:
        raise ValueError(f"Expected [batch,time,classes] CTC output, got {logits.shape}")
    indices = logits.argmax(axis=2)
    probabilities = logits.max(axis=2)
    results: list[tuple[str, float]] = []
    for sequence, scores in zip(indices, probabilities, strict=True):
        text: list[str] = []
        accepted: list[float] = []
        previous = -1
        for token, score in zip(sequence.tolist(), scores.tolist(), strict=True):
            if token != 0 and token != previous and token < len(characters):
                text.append(characters[token])
                accepted.append(float(score))
            previous = token
        results.append(("".join(text), float(np.mean(accepted)) if accepted else 0.0))
    return results
