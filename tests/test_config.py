from pathlib import Path

from basketball_cv.config import PlayerTrackingConfig


def test_player_tracking_config_defaults():
    config = PlayerTrackingConfig(input_video=Path("data/raw_videos/sample.mp4"))

    assert config.model_weights == "yolo11n.pt"
    assert config.tracker == "botsort.yaml"
    assert config.classes == (0,)
