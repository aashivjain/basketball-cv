from dataclasses import dataclass
from pathlib import Path
from typing import Any

from basketball_cv.config import PlayerTrackingConfig
from basketball_cv.detection.yolo import load_yolo_model
from basketball_cv.tracking.track_data import extract_track_points, save_track_points


@dataclass(frozen=True)
class TrackResult:
    output_dir: Path
    results: Any


def track_players(config: PlayerTrackingConfig) -> TrackResult:
    model = load_yolo_model(config.model_weights)
    results = model.track(
        source=str(config.input_video),
        tracker=str(config.tracker),
        classes=list(config.classes),
        conf=config.confidence,
        save=config.save_video,
        persist=config.persist_tracks,
        project=str(config.output_dir),
        name="track_players",
    )
    track_result = TrackResult(output_dir=config.output_dir, results=results)

    if config.save_track_table:
        track_points = extract_track_points(results)
        track_table_path = config.output_dir / config.track_table_name
        save_track_points(track_points, track_table_path)

    return track_result
