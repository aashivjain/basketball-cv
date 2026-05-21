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


def _compute_box_iou(box1: TrackPoint, box2: TrackPoint) -> float:
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    x1_inter = max(box1.x1, box2.x1)
    y1_inter = max(box1.y1, box2.y1)
    x2_inter = min(box1.x2, box2.x2)
    y2_inter = min(box1.y2, box2.y2)
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def _compute_center_distance(box1: TrackPoint, box2: TrackPoint) -> float:
    """Compute Euclidean distance between box centers."""
    cx1, cy1 = box1.center
    cx2, cy2 = box2.center
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def improve_track_consistency(points: list[TrackPoint], iou_threshold: float = 0.3, max_gap_frames: int = 2) -> list[TrackPoint]:
    """Improve track ID consistency by bridging gaps and reducing ID flicker.
    
    Args:
        points: List of TrackPoint objects from raw YOLO tracking
        iou_threshold: IoU threshold for spatial matching (0-1)
        max_gap_frames: Maximum frames to bridge when a player disappears briefly
    
    Returns:
        Improved list of TrackPoints with more stable track IDs
    """
    if not points:
        return points
    
    # Sort by frame then track_id for processing
    sorted_points = sorted(points, key=lambda p: (p.frame, p.track_id or -1))
    
    # Group points by frame
    frame_points: dict[int, list[TrackPoint]] = {}
    for point in sorted_points:
        if point.frame not in frame_points:
            frame_points[point.frame] = []
        frame_points[point.frame].append(point)
    
    frames = sorted(frame_points.keys())
    improved_points: list[TrackPoint] = []
    
    # Process frame by frame, bridging gaps in tracks
    for i, current_frame in enumerate(frames):
        current_detections = frame_points[current_frame]
        
        # Look for untracked detections (track_id is None)
        untracked = [p for p in current_detections if p.track_id is None]
        tracked = [p for p in current_detections if p.track_id is not None]
        
        # Try to match untracked detections to previous tracks
        if untracked and i > 0:
            # Look back up to max_gap_frames frames
            for lookback_offset in range(1, min(max_gap_frames + 1, i + 1)):
                prev_frame = frames[i - lookback_offset]
                prev_detections = frame_points[prev_frame]
                
                for untracked_point in untracked[:]:  # Copy list to modify during iteration
                    best_match = None
                    best_iou = 0
                    
                    for prev_point in prev_detections:
                        if prev_point.track_id is None:
                            continue
                        
                        iou = _compute_box_iou(untracked_point, prev_point)
                        if iou > max(best_iou, iou_threshold):
                            best_iou = iou
                            best_match = prev_point.track_id
                    
                    # If good spatial match found, assign the track ID
                    if best_match is not None:
                        improved_point = TrackPoint(
                            frame=untracked_point.frame,
                            track_id=best_match,
                            class_id=untracked_point.class_id,
                            confidence=untracked_point.confidence,
                            x1=untracked_point.x1,
                            y1=untracked_point.y1,
                            x2=untracked_point.x2,
                            y2=untracked_point.y2,
                        )
                        improved_points.append(improved_point)
                        untracked.remove(untracked_point)
                        break  # Move to next untracked point
        
        # Add remaining tracked points
        improved_points.extend(tracked)
        
        # Add any remaining untracked points (couldn't be matched)
        improved_points.extend(untracked)
    
    return sorted(improved_points, key=lambda p: (p.frame, p.track_id or -1))


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
