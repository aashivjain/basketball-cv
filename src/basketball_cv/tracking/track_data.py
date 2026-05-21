from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    track_id: int | None
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "center_x": self.center[0],
            "center_y": self.center[1],
        }


def _to_float_sequence(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    values.append(float(sub_item))
            else:
                values.append(float(item))
        return tuple(values)
    return (float(value),)


def _resolve_box_coordinates(box: Any) -> tuple[float, float, float, float]:
    xyxy = _to_float_sequence(getattr(box, "xyxy", None))
    if len(xyxy) >= 4:
        return xyxy[0], xyxy[1], xyxy[2], xyxy[3]

    data = _to_float_sequence(getattr(box, "data", None))
    if len(data) >= 4:
        return data[0], data[1], data[2], data[3]

    raise ValueError("Unable to resolve box coordinates from YOLO result.")


def _resolve_class_id(box: Any) -> int:
    cls = getattr(box, "cls", None)
    if cls is not None:
        return int(cls)

    data = _to_float_sequence(getattr(box, "data", None))
    if len(data) >= 6:
        return int(data[5])

    raise ValueError("Unable to resolve class ID from YOLO result.")


def _resolve_confidence(box: Any) -> float:
    conf = getattr(box, "conf", None)
    if conf is not None:
        return float(conf)

    data = _to_float_sequence(getattr(box, "data", None))
    if len(data) >= 5:
        return float(data[4])

    raise ValueError("Unable to resolve confidence score from YOLO result.")


def _resolve_track_id(box: Any) -> int | None:
    track_id = getattr(box, "id", None)
    if track_id is not None:
        return int(track_id)

    data = _to_float_sequence(getattr(box, "data", None))
    if len(data) >= 7:
        return int(data[6])

    return None


def extract_track_points(results: Any) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    if results is None:
        return points

    iterator = results if isinstance(results, (list, tuple)) else list(results)
    for frame_index, result in enumerate(iterator):
        frame_number = int(getattr(result, "frame", frame_index))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = _resolve_box_coordinates(box)
            class_id = _resolve_class_id(box)
            confidence = _resolve_confidence(box)
            track_id = _resolve_track_id(box)
            points.append(
                TrackPoint(
                    frame=frame_number,
                    track_id=track_id,
                    class_id=class_id,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
    return points


def save_track_points(points: Iterable[TrackPoint], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "track_id",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "center_x",
        "center_y",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point.to_dict())
    return output_path


def convert_video_format(input_path: Path, output_path: Path, target_format: str = "mp4") -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if target_format.lower() == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif target_format.lower() == "avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return output_path
