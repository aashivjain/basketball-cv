from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CourtCalibration:
    """Calibration points mapping video pixels to court feet coordinates."""
    video_points: list[list[float]]  # [[px_x, px_y], ...]
    court_points: list[list[float]]  # [[ft_x, ft_y], ...]

    def is_valid(self) -> bool:
        return (
            len(self.video_points) >= 4
            and len(self.video_points) == len(self.court_points)
        )


@dataclass
class PlayerTrackingConfig:
    input_video: Path
    output_dir: Path = Path("outputs")
    model_weights: str = "yolo11n.pt"
    tracker: str = "botsort.yaml"
    classes: tuple[int, ...] = (0,)
    confidence: float = 0.25
    save_video: bool = True
    persist_tracks: bool = True
    save_track_table: bool = True
    track_table_name: str = "player_tracks.csv"
    video_output_format: str = "mp4"
    court_calibration: CourtCalibration | None = None

    def validate(self) -> None:
        if not self.input_video.exists():
            raise FileNotFoundError(f"Video file not found: {self.input_video}")
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_player_tracking_config(path: str | Path) -> PlayerTrackingConfig:
    raw_config = load_yaml(path)

    # Parse court calibration if present
    court_cal = None
    cal_raw = raw_config.get("court_calibration")
    if cal_raw and "video_points" in cal_raw and "court_points" in cal_raw:
        court_cal = CourtCalibration(
            video_points=cal_raw["video_points"],
            court_points=cal_raw["court_points"],
        )

    return PlayerTrackingConfig(
        input_video=Path(raw_config["input_video"]),
        output_dir=Path(raw_config.get("output_dir", "outputs")),
        model_weights=raw_config.get("model_weights", "yolo11n.pt"),
        tracker=raw_config.get("tracker", "botsort.yaml"),
        classes=tuple(int(value) for value in raw_config.get("classes", [0])),
        confidence=float(raw_config.get("confidence", 0.25)),
        save_video=bool(raw_config.get("save_video", True)),
        persist_tracks=bool(raw_config.get("persist_tracks", True)),
        save_track_table=bool(raw_config.get("save_track_table", True)),
        track_table_name=str(raw_config.get("track_table_name", "player_tracks.csv")),
        video_output_format=str(raw_config.get("video_output_format", "mp4")),
        court_calibration=court_cal,
    )
