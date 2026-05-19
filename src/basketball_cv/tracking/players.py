from basketball_cv.config import PlayerTrackingConfig
from basketball_cv.detection.yolo import load_yolo_model


def track_players(config: PlayerTrackingConfig):
    model = load_yolo_model(config.model_weights)

    return model.track(
        source=str(config.input_video),
        tracker=config.tracker,
        classes=list(config.classes),
        conf=config.confidence,
        save=config.save_video,
        persist=config.persist_tracks,
    )
