from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


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


def split_merged_detections(points: list[TrackPoint]) -> list[TrackPoint]:
    """Split bounding boxes that are likely merging multiple players.
    
    If a detection is significantly wider than normal player width, 
    split it into multiple detections.
    
    Args:
        points: Raw track points from YOLO
        
    Returns:
        Points with merged detections split into individual player boxes
    """
    if not points:
        return points

    # Compute median width from all detections to establish "normal" player size
    widths = [p.width for p in points if p.width > 10]
    if not widths:
        return points
    
    widths_sorted = sorted(widths)
    median_width = widths_sorted[len(widths_sorted) // 2]
    
    # Also compute median height for aspect ratio check
    heights = [p.height for p in points if p.height > 10]
    median_height = sorted(heights)[len(heights) // 2] if heights else 100

    # Threshold: if width is > 1.7x median, likely merged
    split_threshold = median_width * 1.7
    
    result = []
    next_split_id = max((p.track_id or 0) for p in points) + 1000  # Avoid ID conflicts
    
    for point in points:
        if point.width <= split_threshold:
            result.append(point)
            continue
        
        # This box is too wide - split it
        n_players = max(2, round(point.width / median_width))
        split_width = point.width / n_players
        
        for j in range(n_players):
            new_x1 = point.x1 + j * split_width
            new_x2 = new_x1 + split_width
            
            # First split keeps original track_id, others get new IDs
            tid = point.track_id if j == 0 else None
            
            result.append(TrackPoint(
                frame=point.frame,
                track_id=tid,
                class_id=point.class_id,
                confidence=point.confidence * 0.9,  # Slightly lower confidence for splits
                x1=new_x1,
                y1=point.y1,
                x2=new_x2,
                y2=point.y2,
            ))
    
    return result


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


def improve_track_consistency_with_appearance(
    points: list[TrackPoint], 
    frames_data: dict[int, np.ndarray],
    iou_threshold: float = 0.3,
    max_gap_frames: int = 5,
    appearance_threshold: float = 0.6
) -> list[TrackPoint]:
    """Improve tracking using both spatial and appearance features.
    
    Handles two cases:
    1. Untracked detections (track_id=None) - match to previous tracks
    2. New IDs that appear near where an old ID disappeared - reassign to old ID
    
    Args:
        points: List of TrackPoint objects
        frames_data: Dict mapping frame number to frame image (numpy array)
        iou_threshold: IoU threshold for spatial matching
        max_gap_frames: Max frames to bridge for disappeared tracks
        appearance_threshold: Appearance distance threshold (lower = stricter)
    
    Returns:
        Improved track points with stable IDs across crossings
    """
    if not points:
        return points
    
    try:
        from basketball_cv.appearance import extract_appearance, compute_appearance_distance
    except ImportError:
        return improve_track_consistency(points, iou_threshold, max_gap_frames)
    
    sorted_points = sorted(points, key=lambda p: (p.frame, p.track_id or -1))
    
    # Group points by frame
    frame_points: dict[int, list[TrackPoint]] = {}
    for point in sorted_points:
        if point.frame not in frame_points:
            frame_points[point.frame] = []
        frame_points[point.frame].append(point)
    
    frames = sorted(frame_points.keys())
    
    # --- Phase 1: Build track history and detect ID switches ---
    # Track when each ID was last seen and where
    track_last_seen: dict[int, tuple[int, TrackPoint]] = {}  # track_id -> (frame, point)
    track_first_seen: dict[int, int] = {}  # track_id -> first frame
    
    for frame_num in frames:
        for pt in frame_points[frame_num]:
            if pt.track_id is not None:
                track_last_seen[pt.track_id] = (frame_num, pt)
                if pt.track_id not in track_first_seen:
                    track_first_seen[pt.track_id] = frame_num
    
    # --- Phase 2: Identify ID reassignments ---
    # A new ID appearing where an old ID just disappeared is likely the same player
    id_mapping: dict[int, int] = {}  # new_id -> should_be_id
    
    # Cache appearance features
    appearance_cache: dict[tuple[int, int, int, int], Any] = {}
    
    def get_appearance_for_point(frame_num: int, pt: TrackPoint):
        key = (frame_num, int(pt.x1), int(pt.y1), int(pt.x2))
        if key not in appearance_cache:
            if frame_num not in frames_data:
                return None
            try:
                feat = extract_appearance(frames_data[frame_num], (pt.x1, pt.y1, pt.x2, pt.y2))
                appearance_cache[key] = feat
            except Exception:
                return None
        return appearance_cache.get(key)
    
    # For each frame, check if new IDs match recently-disappeared old IDs
    active_tracks: dict[int, tuple[int, TrackPoint]] = {}  # id -> (last_frame, last_point)
    
    for i, frame_num in enumerate(frames):
        current_ids = set()
        current_points_by_id: dict[int, TrackPoint] = {}
        
        for pt in frame_points[frame_num]:
            if pt.track_id is not None:
                current_ids.add(pt.track_id)
                current_points_by_id[pt.track_id] = pt
        
        # Find new IDs in this frame (first appearance)
        new_ids_this_frame = [
            tid for tid in current_ids 
            if track_first_seen.get(tid) == frame_num
        ]
        
        # Find recently disappeared IDs
        disappeared_ids = {}
        for tid, (last_frame, last_pt) in active_tracks.items():
            if tid not in current_ids and (frame_num - last_frame) <= max_gap_frames:
                disappeared_ids[tid] = (last_frame, last_pt)
        
        # Try to match new IDs to disappeared IDs
        for new_id in new_ids_this_frame:
            if new_id in id_mapping:
                continue
            
            new_pt = current_points_by_id[new_id]
            best_match_id = None
            best_score = 0
            
            for old_id, (old_frame, old_pt) in disappeared_ids.items():
                if old_id in id_mapping.values():
                    continue  # Already remapped
                
                # Spatial proximity (center distance)
                dist = _compute_center_distance(new_pt, old_pt)
                frame_gap = frame_num - old_frame
                # Allow more distance for larger gaps (player moved)
                max_dist = 80 + frame_gap * 40
                
                if dist > max_dist:
                    continue
                
                # IoU check (may be 0 if player moved, so use distance as primary)
                iou = _compute_box_iou(new_pt, old_pt)
                
                # Appearance check
                feat_new = get_appearance_for_point(frame_num, new_pt)
                feat_old = get_appearance_for_point(old_frame, old_pt)
                
                appearance_sim = 0.5
                if feat_new and feat_old:
                    app_dist = compute_appearance_distance(feat_new, feat_old)
                    appearance_sim = 1.0 - min(1.0, app_dist)
                
                # Combined score: spatial proximity + appearance + IoU
                proximity_score = max(0, 1.0 - dist / max_dist)
                combined = 0.4 * appearance_sim + 0.35 * proximity_score + 0.25 * iou
                
                if combined > best_score and combined > 0.35:
                    best_score = combined
                    best_match_id = old_id
            
            if best_match_id is not None:
                id_mapping[new_id] = best_match_id
        
        # Update active tracks
        for tid, pt in current_points_by_id.items():
            active_tracks[tid] = (frame_num, pt)
    
    # --- Phase 3: Apply ID mapping and handle untracked points ---
    improved_points: list[TrackPoint] = []
    
    for i, frame_num in enumerate(frames):
        current_detections = frame_points[frame_num]
        
        for pt in current_detections:
            if pt.track_id is None:
                # Try to match untracked to nearby recent tracks
                best_match = None
                best_score = 0
                
                for lookback in range(1, min(max_gap_frames + 1, i + 1)):
                    prev_frame = frames[i - lookback]
                    for prev_pt in frame_points[prev_frame]:
                        if prev_pt.track_id is None:
                            continue
                        
                        dist = _compute_center_distance(pt, prev_pt)
                        if dist > 120:
                            continue
                        
                        iou = _compute_box_iou(pt, prev_pt)
                        feat_cur = get_appearance_for_point(frame_num, pt)
                        feat_prev = get_appearance_for_point(prev_frame, prev_pt)
                        
                        appearance_sim = 0.5
                        if feat_cur and feat_prev:
                            app_dist = compute_appearance_distance(feat_cur, feat_prev)
                            appearance_sim = 1.0 - min(1.0, app_dist)
                        
                        score = 0.5 * appearance_sim + 0.3 * iou + 0.2 * max(0, 1 - dist / 120)
                        if score > best_score and score > 0.35:
                            best_score = score
                            # Apply ID mapping if the matched ID was remapped
                            matched_id = prev_pt.track_id
                            best_match = id_mapping.get(matched_id, matched_id)
                    
                    if best_match is not None:
                        break
                
                improved_points.append(TrackPoint(
                    frame=pt.frame,
                    track_id=best_match,
                    class_id=pt.class_id,
                    confidence=pt.confidence,
                    x1=pt.x1, y1=pt.y1, x2=pt.x2, y2=pt.y2,
                ))
            else:
                # Apply ID mapping for reassigned IDs
                mapped_id = id_mapping.get(pt.track_id, pt.track_id)
                if mapped_id != pt.track_id:
                    improved_points.append(TrackPoint(
                        frame=pt.frame,
                        track_id=mapped_id,
                        class_id=pt.class_id,
                        confidence=pt.confidence,
                        x1=pt.x1, y1=pt.y1, x2=pt.x2, y2=pt.y2,
                    ))
                else:
                    improved_points.append(pt)
    
    return sorted(improved_points, key=lambda p: (p.frame, p.track_id or -1))


def consolidate_track_ids(
    points: list[TrackPoint],
    frames_data: dict[int, np.ndarray],
    max_gap_frames: int = 10,
    max_distance: float = 150.0,
) -> list[TrackPoint]:
    """Consolidate fragmented track IDs that belong to the same player.
    
    If two track IDs never co-exist in the same frame AND are spatially/appearance
    similar when one ends and the other begins, merge them into one ID.
    
    Args:
        points: Track points (already improved by appearance matching)
        frames_data: Dict of frame images
        max_gap_frames: Maximum frame gap to consider for merging
        max_distance: Maximum center distance to consider merging
    
    Returns:
        Points with consolidated IDs (fewer unique IDs)
    """
    if not points:
        return points
    
    try:
        from basketball_cv.appearance import extract_appearance, compute_appearance_distance
    except ImportError:
        return points
    
    # Build per-track info: frame range, positions, appearance
    track_info: dict[int, dict] = {}  # track_id -> {first_frame, last_frame, points, ...}
    
    for pt in points:
        if pt.track_id is None:
            continue
        tid = pt.track_id
        if tid not in track_info:
            track_info[tid] = {
                'first_frame': pt.frame,
                'last_frame': pt.frame,
                'first_point': pt,
                'last_point': pt,
                'count': 0,
            }
        info = track_info[tid]
        info['count'] += 1
        if pt.frame < info['first_frame']:
            info['first_frame'] = pt.frame
            info['first_point'] = pt
        if pt.frame >= info['last_frame']:
            info['last_frame'] = pt.frame
            info['last_point'] = pt
    
    # Find which tracks co-exist (appear in same frame)
    frame_ids: dict[int, set[int]] = {}
    for pt in points:
        if pt.track_id is None:
            continue
        if pt.frame not in frame_ids:
            frame_ids[pt.frame] = set()
        frame_ids[pt.frame].add(pt.track_id)
    
    coexist: set[tuple[int, int]] = set()
    for frame_num, ids in frame_ids.items():
        ids_list = list(ids)
        for a in range(len(ids_list)):
            for b in range(a + 1, len(ids_list)):
                pair = (min(ids_list[a], ids_list[b]), max(ids_list[a], ids_list[b]))
                coexist.add(pair)
    
    # Build merge candidates: tracks that DON'T co-exist and are temporally close
    all_ids = sorted(track_info.keys())
    merge_map: dict[int, int] = {}  # new_id -> canonical_id
    
    # Sort tracks by first appearance
    sorted_ids = sorted(all_ids, key=lambda tid: track_info[tid]['first_frame'])
    
    for i, tid_b in enumerate(sorted_ids):
        if tid_b in merge_map:
            continue
        
        info_b = track_info[tid_b]
        best_merge = None
        best_score = 0
        
        # Look for earlier tracks that ended before this one started
        for tid_a in sorted_ids[:i]:
            # Get canonical ID for tid_a
            canonical_a = tid_a
            while canonical_a in merge_map:
                canonical_a = merge_map[canonical_a]
            
            info_a = track_info[tid_a]
            
            # Check temporal constraint: tid_a ended before tid_b started (with gap)
            gap = info_b['first_frame'] - info_a['last_frame']
            if gap < 0 or gap > max_gap_frames:
                continue
            
            # Check they don't co-exist
            pair = (min(canonical_a, tid_b), max(canonical_a, tid_b))
            if pair in coexist:
                continue
            
            # Check spatial proximity
            dist = _compute_center_distance(info_a['last_point'], info_b['first_point'])
            if dist > max_distance:
                continue
            
            # Appearance check
            feat_a = None
            feat_b = None
            if info_a['last_frame'] in frames_data:
                try:
                    pt_a = info_a['last_point']
                    feat_a = extract_appearance(
                        frames_data[info_a['last_frame']], 
                        (pt_a.x1, pt_a.y1, pt_a.x2, pt_a.y2)
                    )
                except Exception:
                    pass
            if info_b['first_frame'] in frames_data:
                try:
                    pt_b = info_b['first_point']
                    feat_b = extract_appearance(
                        frames_data[info_b['first_frame']], 
                        (pt_b.x1, pt_b.y1, pt_b.x2, pt_b.y2)
                    )
                except Exception:
                    pass
            
            appearance_sim = 0.5
            if feat_a and feat_b:
                app_dist = compute_appearance_distance(feat_a, feat_b)
                appearance_sim = 1.0 - min(1.0, app_dist)
            
            # Score based on proximity, appearance, and gap
            proximity_score = max(0, 1.0 - dist / max_distance)
            gap_score = max(0, 1.0 - gap / max_gap_frames)
            combined = 0.4 * appearance_sim + 0.35 * proximity_score + 0.25 * gap_score
            
            if combined > best_score and combined > 0.4:
                best_score = combined
                best_merge = canonical_a
        
        if best_merge is not None:
            merge_map[tid_b] = best_merge
    
    # Resolve transitive merges: if A->B and B->C, then A->C
    def resolve(tid):
        visited = set()
        while tid in merge_map and tid not in visited:
            visited.add(tid)
            tid = merge_map[tid]
        return tid
    
    # Apply merges
    result = []
    for pt in points:
        if pt.track_id is not None and pt.track_id in merge_map:
            canonical = resolve(pt.track_id)
            result.append(TrackPoint(
                frame=pt.frame,
                track_id=canonical,
                class_id=pt.class_id,
                confidence=pt.confidence,
                x1=pt.x1, y1=pt.y1, x2=pt.x2, y2=pt.y2,
            ))
        else:
            result.append(pt)
    
    return sorted(result, key=lambda p: (p.frame, p.track_id or -1))


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


def filter_crowd_and_low_confidence(
    points: list[TrackPoint],
    frames_data: dict[int, np.ndarray],
    confidence_threshold: float = 0.15,
    filter_crowd: bool = True
) -> list[TrackPoint]:
    """Filter out crowd members and low-confidence detections.
    
    Args:
        points: List of track points
        frames_data: Dict mapping frame number to frame image
        confidence_threshold: Minimum confidence to keep (0-1)
        filter_crowd: Whether to apply crowd filtering
        
    Returns:
        Filtered list of track points
    """
    if not filter_crowd:
        # Just filter by confidence
        return [p for p in points if p.confidence >= confidence_threshold]
    
    try:
        from basketball_cv.appearance import filter_crowd_detections
    except ImportError:
        # Fallback to confidence-only filtering
        return [p for p in points if p.confidence >= confidence_threshold]
    
    filtered = []
    
    # Group by frame for batch processing
    frame_points: dict[int, list[TrackPoint]] = {}
    for point in points:
        if point.frame not in frame_points:
            frame_points[point.frame] = []
        frame_points[point.frame].append(point)
    
    # Filter each frame
    for frame_num, points_in_frame in frame_points.items():
        if frame_num not in frames_data:
            # Can't filter without frame data, keep all
            filtered.extend(points_in_frame)
            continue
        
        frame = frames_data[frame_num]
        
        # Get bounding boxes
        bboxes = [(p.x1, p.y1, p.x2, p.y2) for p in points_in_frame]
        
        try:
            # Identify crowd members
            crowd_indices = filter_crowd_detections(frame, bboxes)
            crowd_set = set(crowd_indices)
        except Exception:
            crowd_set = set()
        
        # Keep points that pass confidence and are not crowd
        for idx, point in enumerate(points_in_frame):
            if point.confidence >= confidence_threshold and idx not in crowd_set:
                filtered.append(point)
    
    return filtered


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
