from dataclasses import dataclass
from pathlib import Path
from typing import Any

from basketball_cv.config import PlayerTrackingConfig
from basketball_cv.detection.yolo import load_yolo_model
from basketball_cv.tracking.track_data import extract_track_points, save_track_points, convert_video_format


@dataclass(frozen=True)
class TrackResult:
    output_dir: Path
    results: Any


def _find_latest_yolo_output(base_dir: Path) -> Path | None:
    detect_dir = base_dir / "detect" / "outputs"
    if not detect_dir.exists():
        return None
    track_dirs = sorted([d for d in detect_dir.iterdir() if d.is_dir() and d.name.startswith("track_players")])
    return track_dirs[-1] if track_dirs else None


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

    if config.save_video and config.video_output_format.lower() == "mp4":
        yolo_output_dir = _find_latest_yolo_output(Path("runs"))
        if yolo_output_dir:
            avi_path = yolo_output_dir / "sample.avi"
            mp4_path = yolo_output_dir / "sample.mp4"
            if avi_path.exists():
                convert_video_format(avi_path, mp4_path, target_format="mp4")
                avi_path.unlink()

    if config.save_track_table:
        track_points = extract_track_points(results)
        track_table_path = config.output_dir / config.track_table_name
        save_track_points(track_points, track_table_path)

    return track_result
