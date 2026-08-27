from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .domain import ObjectDetection
from .interfaces import ObjectDetector


COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

ZH_LABELS = {
    "person": "人物", "bicycle": "自行車", "car": "汽車", "motorcycle": "機車", "bus": "公車",
    "train": "火車", "truck": "卡車", "boat": "船", "bird": "鳥", "cat": "貓", "dog": "狗",
    "horse": "馬", "sheep": "羊", "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑馬",
    "giraffe": "長頸鹿", "backpack": "背包", "umbrella": "雨傘", "handbag": "手提包",
    "suitcase": "行李箱", "bottle": "瓶子", "cup": "杯子", "chair": "椅子", "couch": "沙發",
    "bed": "床", "dining table": "桌子", "tv": "電視", "laptop": "筆電", "mouse": "滑鼠",
    "keyboard": "鍵盤", "cell phone": "手機", "book": "書", "clock": "時鐘", "scissors": "剪刀",
    "multimeter": "萬用電表", "oscilloscope": "示波器", "soldering iron": "烙鐵",
}


def _load_metadata(model_dir: Path) -> tuple[list[str], dict[str, object]]:
    metadata: dict[str, object] = {}
    yaml_path = model_dir / "metadata.yaml"
    if yaml_path.is_file():
        metadata.update(yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {})
    config_path = model_dir / "openvino_config.json"
    if config_path.is_file():
        metadata.update(json.loads(config_path.read_text(encoding="utf-8")))
    vocabulary_path = model_dir / str(metadata.get("vocabulary_file", "vocabulary.json"))
    names = metadata.get("names", [])
    if vocabulary_path.is_file():
        payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        names = payload.get("labels", payload.get("names", payload)) if isinstance(payload, dict) else payload
    if isinstance(names, dict):
        ordered = [str(value) for _key, value in sorted(names.items(), key=lambda item: int(item[0]))]
    elif isinstance(names, list):
        ordered = [str(value) for value in names]
    else:
        ordered = []
    return ordered, metadata


class NoopObjectDetector(ObjectDetector):
    model_version = "objects-disabled"

    def detect(self, image: Path) -> list[ObjectDetection]:
        return []

    def runtime_details(self) -> list[dict[str, object]]:
        return [{"name": "物件辨識", "backend": "未安裝", "execution_devices": [], "loaded": False}]


class OpenVinoYoloEngine(ObjectDetector):
    """Native OpenVINO YOLOE-26 prompt-free detector without PyTorch."""

    def __init__(self, model_dir: Path, device: str = "GPU", confidence: float = 0.20,
                 iou_threshold: float = 0.70, max_detections: int = 50):
        import openvino as ov
        from .engines import compiled_execution_devices, safe_compiled_property, select_openvino_device, serial_inference_config

        self.model_dir = Path(model_dir)
        model_path = next(iter(sorted(self.model_dir.glob("*.xml"))), None)
        if model_path is None:
            raise FileNotFoundError("YOLOE-26 OpenVINO XML not found")
        self.labels, self.metadata = _load_metadata(self.model_dir)
        self.scale = str(self.metadata.get("scale", "S")).upper()
        self.model_version = str(
            self.metadata.get("model_version", f"yoloe-26{self.scale.casefold()}-pf-openvino-fp16-v1")
        )
        core = ov.Core()
        self.device = select_openvino_device(core, device)
        cache = self.model_dir / "compiled_cache"
        cache.mkdir(exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache)})
        self.compiled = core.compile_model(model_path, self.device, serial_inference_config(self.device))
        self.request = self.compiled.create_infer_request()
        self.optimal_infer_requests = safe_compiled_property(
            self.compiled, "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
        )
        self.execution_devices = compiled_execution_devices(self.compiled, self.device)
        shape = self.compiled.input(0).shape
        self.input_height, self.input_width = int(shape[-2]), int(shape[-1])
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max(1, int(max_detections))

    def detect(self, image: Path) -> list[ObjectDetection]:
        data = np.fromfile(image, dtype=np.uint8)
        source = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"Unable to decode image: {image}")
        tensor, scale, pad_x, pad_y = letterbox(source, self.input_width, self.input_height)
        self.request.infer({self.compiled.input(0): tensor})
        outputs = [
            np.array(self.request.get_output_tensor(index).data, copy=True)
            for index, _port in enumerate(self.compiled.outputs)
        ]
        boxes, scores, classes = parse_yoloe_outputs(outputs, self.confidence, len(self.labels))
        if not len(boxes):
            return []
        visible = visible_box_mask(
            boxes,
            pad_x,
            pad_y,
            source.shape[1] * scale,
            source.shape[0] * scale,
        )
        boxes, scores, classes = boxes[visible], scores[visible], classes[visible]
        if not len(boxes):
            return []
        nms = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence, self.iou_threshold)
        indices = np.asarray(nms).reshape(-1).tolist() if len(nms) else []
        indices = sorted(indices, key=lambda index: float(scores[index]), reverse=True)[:self.max_detections]
        height, width = source.shape[:2]
        result: list[ObjectDetection] = []
        for index in indices:
            x, y, w, h = boxes[index]
            x1 = max(0.0, min(width, (x - pad_x) / scale))
            y1 = max(0.0, min(height, (y - pad_y) / scale))
            x2 = max(0.0, min(width, (x + w - pad_x) / scale))
            y2 = max(0.0, min(height, (y + h - pad_y) / scale))
            class_id = int(classes[index])
            label = self.labels[class_id] if 0 <= class_id < len(self.labels) else f"class-{class_id}"
            result.append(ObjectDetection(
                label, float(scores[index]), [float(x1), float(y1), float(x2), float(y2)],
                ZH_LABELS.get(label.casefold(), "")
            ))
        return result

    def smoke_test(self) -> None:
        shape = self.compiled.input(0).shape
        tensor = np.zeros(tuple(int(item) for item in shape), dtype=np.float32)
        self.request.infer({self.compiled.input(0): tensor})
        if not all(
            np.isfinite(np.asarray(self.request.get_output_tensor(index).data)).all()
            for index, _port in enumerate(self.compiled.outputs)
        ):
            raise RuntimeError("YOLOE-26 OpenVINO smoke test failed")

    def runtime_details(self) -> list[dict[str, object]]:
        return [{
            "name": f"YOLOE-26{self.scale} 開放物件辨識",
            "backend": "OpenVINO FP16", "requested_device": self.device,
            "execution_devices": self.execution_devices, "selected_device": self.device, "loaded": True,
            "performance_hint": "LATENCY",
            "optimal_infer_requests": self.optimal_infer_requests,
            "prompt_mode": "prompt-free", "vocabulary_size": len(self.labels), "scale": self.scale,
        }]


def letterbox(source: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    source_h, source_w = source.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w, resized_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    resized = cv2.resize(source, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (width - resized_w) // 2, (height - resized_h) // 2
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, pad_x, pad_y


def visible_box_mask(boxes: np.ndarray, pad_x: float, pad_y: float,
                     content_width: float, content_height: float,
                     minimum_size: float = 2.0) -> np.ndarray:
    """Reject padded detections that do not cover real letterboxed pixels."""
    if not len(boxes):
        return np.zeros((0,), dtype=bool)
    x1 = np.clip(boxes[:, 0], pad_x, pad_x + content_width)
    y1 = np.clip(boxes[:, 1], pad_y, pad_y + content_height)
    x2 = np.clip(boxes[:, 0] + boxes[:, 2], pad_x, pad_x + content_width)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3], pad_y, pad_y + content_height)
    return ((x2 - x1) >= minimum_size) & ((y2 - y1) >= minimum_size)


def parse_yoloe_outputs(outputs: list[np.ndarray], threshold: float,
                        class_count: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.asarray(item) for item in outputs]
    box_tensor = next((item for item in arrays if item.ndim >= 2 and item.shape[-1] == 4), None)
    if box_tensor is not None:
        count = int(np.asarray(box_tensor).reshape(-1, 4).shape[0])
        vectors = [np.asarray(item).reshape(-1) for item in arrays if item is not box_tensor]
        score_tensor = next((item for item in vectors if item.size == count and np.issubdtype(item.dtype, np.floating)), None)
        class_tensor = next((item for item in vectors if item.size == count and np.issubdtype(item.dtype, np.integer)), None)
        if score_tensor is not None and class_tensor is not None:
            xyxy = np.asarray(box_tensor, dtype=np.float32).reshape(-1, 4)
            scores = score_tensor.astype(np.float32)
            classes = class_tensor.astype(np.int32)
            boxes = np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]))
            finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
            has_area = (boxes[:, 2] >= 2.0) & (boxes[:, 3] >= 2.0)
            keep = (scores >= threshold) & finite & has_area
            return boxes[keep], scores[keep], classes[keep]
    candidates = [item for item in arrays if item.ndim in {2, 3}]
    if not candidates:
        raise ValueError("YOLOE output has no detection tensor")
    output = next((item for item in candidates if 6 in item.shape[-2:]), candidates[0])
    return parse_yolo_output(output, threshold, class_count)


def parse_yolo_output(output: np.ndarray, threshold: float,
                      class_count: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.squeeze(output)
    if values.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {output.shape}")
    # Raw YOLO exports are channel-first (channels, anchors); end-to-end NMS
    # exports are row-first (detections, 6 + optional mask coefficients).
    known_channel_first = values.shape[0] in {6, 84, 85} and values.shape[1] not in {6, 84, 85}
    if known_channel_first or (
        values.shape[0] < values.shape[1] and values.shape[0] <= max(512, class_count + 68)
    ):
        values = values.T
    known_raw_width = values.shape[1] in {84, 85}
    looks_like_nms = not known_raw_width and values.shape[1] >= 6 and (
        values.shape[1] == 6
        or (class_count > 0 and values.shape[1] < 4 + class_count)
        or (
            np.all((values[:, 4] >= 0) & (values[:, 4] <= 1))
            and np.mean(np.isclose(values[:, 5], np.round(values[:, 5]), atol=1e-4)) > 0.95
        )
    )
    if looks_like_nms:
        scores = values[:, 4]
        classes = values[:, 5].astype(np.int32)
        xyxy = values[:, :4]
        boxes = np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]))
    else:
        available = values.shape[1] - 4
        score_count = class_count if 0 < class_count <= available else available
        class_scores = values[:, 4:4 + score_count]
        classes = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(len(values)), classes]
        center = values[:, :4]
        boxes = np.column_stack((center[:, 0] - center[:, 2] / 2, center[:, 1] - center[:, 3] / 2,
                                 center[:, 2], center[:, 3]))
    # Prompt-free end-to-end exports pad Top-K rows with zero-area boxes. The
    # padding can retain a high sentinel score/class id after FP16 conversion,
    # so confidence alone is not sufficient to identify a real detection.
    finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
    has_area = (boxes[:, 2] >= 2.0) & (boxes[:, 3] >= 2.0)
    keep = (scores >= threshold) & finite & has_area
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32), classes[keep]
