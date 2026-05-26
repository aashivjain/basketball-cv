from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from basketball_cv.config import PlayerTrackingConfig
from basketball_cv.court import CourtMapper
from basketball_cv.detection.yolo import load_yolo_model
from basketball_cv.tracking.track_data import (
    extract_track_points,
    save_track_points,
)
from basketball_cv.rendering import render_tracked_video


@dataclass(frozen=True)
class TrackResult:
    output_dir: Path
    results: Any


def track_players(config: PlayerTrackingConfig) -> TrackResult:
    """Run player tracking pipeline.
    
    Uses YOLO + BoTSORT with default settings for reliable tracking.
    No post-processing heuristics — trust the tracker's native IDs.
    """
    model = load_yolo_model(config.model_weights)
    results = model.track(
        source=str(config.input_video),
        tracker=str(config.tracker),
        classes=list(config.classes),
        conf=config.confidence,
        save=False,  # We render our own video with court overlay
        persist=config.persist_tracks,
        project=str(config.output_dir),
        name="track_players",
    )
    track_result = TrackResult(output_dir=config.output_dir, results=results)

    if config.save_track_table:
        track_points = extract_track_points(results)
        
        # Save raw tracking data
        track_table_path = config.output_dir / config.track_table_name
        save_track_points(track_points, track_table_path)

        # Build court mapper (auto-detects paint per-frame)
        court_mapper = CourtMapper()
        print("Court mapper initialized (per-frame paint detection).")

        # Render video with bounding boxes and court overlay
        if config.save_video:
            rendered_video_path = config.output_dir / "tracked_with_court.mp4"
            render_tracked_video(
                input_video=config.input_video,
                output_path=rendered_video_path,
                track_points=track_points,
                court_mapper=court_mapper,
            )
            print(f"Rendered video: {rendered_video_path}")

    return track_result
