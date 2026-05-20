from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    )
