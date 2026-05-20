from pathlib import Path

import pytest

from basketball_cv.config import PlayerTrackingConfig, load_player_tracking_config


def test_player_tracking_config_defaults():
    config = PlayerTrackingConfig(input_video=Path("data/raw_videos/sample.mp4"))

    assert config.model_weights == "yolo11n.pt"
    assert config.tracker == "botsort.yaml"
    assert config.classes == (0,)
    assert config.output_dir == Path("outputs")


def test_load_player_tracking_config_from_yaml(tmp_path: Path):
    config_file = tmp_path / "player_tracking.yaml"
    config_file.write_text(
        """input_video: data/raw_videos/sample.mp4\noutput_dir: outputs/tracking\nclasses:\n  - 0\n"""
    )

    config = load_player_tracking_config(config_file)

    assert config.input_video == Path("data/raw_videos/sample.mp4")
    assert config.output_dir == Path("outputs/tracking")
    assert config.classes == (0,)


def test_player_tracking_config_validation_fails_for_missing_video(tmp_path: Path):
    config = PlayerTrackingConfig(input_video=tmp_path / "missing.mp4")

    with pytest.raises(FileNotFoundError):
        config.validate()
