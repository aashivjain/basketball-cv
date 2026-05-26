"""Ball detection and tracking.

Runs a second YOLO pass (class 32 = sports ball) over the video, keeps the
best-confidence hit per frame, removes physically impossible jumps, then
fills remaining gaps via linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO class index for "sports ball"
_BALL_CLASS_ID = 32

# How far (px) the ball centre is allowed to move between frames.
# A hard pass at 30 fps typically travels ~25-30 px; 40 gives some headroom.
_MAX_PX_PER_FRAME = 40

# A valid in-play ball should usually be close to at least one player's box.
_MAX_PLAYER_ASSOC_DIST = 135.0

# If a detection is far from all players, only keep it when confidence is high.
_MIN_UNSUPPORTED_CONF = 0.55

# Only fill gaps shorter than this. A ball that vanishes for more than
# ~12 frames has likely changed hands or moved off-frame; connecting two
# distant detections with a straight line would look worse than no ball.
_MAX_INTERPOLATION_GAP = 12


@dataclass
class BallDetection:
    """Ball position for a single video frame."""

    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    interpolated: bool = False

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def radius(self) -> float:
        return max((self.x2 - self.x1), (self.y2 - self.y1)) / 2.0


class BallTracker:
    """Handles the full ball tracking pipeline for a single video.

    Call ``track()`` once per video — it runs YOLO ball detection, filters
    out noisy detections by checking inter-frame movement, then interpolates
    the remaining gaps so you get a smooth position for almost every frame.
    """

    def __init__(self, confidence: float = 0.1) -> None:
        # Keep the threshold low — COCO ball detection is sketchy on broadcast
        # footage, so we'd rather over-detect and filter than miss the ball.
        self.confidence = confidence

    def track(
        self,
        model: object,
        video_path: str | Path,
        player_centers_by_frame: dict[int, list[tuple[float, float]]] | None = None,
    ) -> list[BallDetection | None]:
        """Return a per-frame list of ball positions for *video_path*.

        Entries are ``None`` for frames where no position could be
        established (before the first detection or after the last).
        """
        raw = self._detect(model, video_path)
        filtered = self._filter_by_motion(raw)
        filtered = self._filter_by_player_context(filtered, player_centers_by_frame)
        return self._interpolate(filtered)

    def _detect(
        self, model: object, video_path: str | Path
    ) -> list[BallDetection | None]:
        """Predict on the full video and return the top detection per frame."""
        detections: list[BallDetection | None] = []
        frame_idx = 0

        for result in model.predict(
            source=str(video_path),
            classes=[_BALL_CLASS_ID],
            conf=self.confidence,
            stream=True,
            verbose=False,
        ):
            best: BallDetection | None = None

            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    conf = float(box.conf)
                    if best is None or conf > best.confidence:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        best = BallDetection(
                            frame=frame_idx,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=conf,
                        )

            detections.append(best)
            frame_idx += 1

        return detections

    def _filter_by_motion(
        self, detections: list[BallDetection | None]
    ) -> list[BallDetection | None]:
        """Discard detections that jump too far from the previous known position."""
        result: list[BallDetection | None] = list(detections)
        last_valid_idx = -1

        for i, det in enumerate(result):
            if det is None:
                continue

            if last_valid_idx == -1:
                last_valid_idx = i
                continue

            prev = result[last_valid_idx]
            gap = i - last_valid_idx
            max_dist = _MAX_PX_PER_FRAME * gap

            dist = float(
                np.hypot(
                    det.center_x - prev.center_x,
                    det.center_y - prev.center_y,
                )
            )
            if dist > max_dist:
                result[i] = None
            else:
                last_valid_idx = i

        return result

    def _interpolate(
        self, detections: list[BallDetection | None]
    ) -> list[BallDetection | None]:
        """Fill gaps between detections using linear interpolation.

        Gaps longer than ``_MAX_INTERPOLATION_GAP`` frames are left as None
        rather than drawing a fake path between distant detections.
        """
        result = list(detections)
        n = len(detections)
        if n == 0:
            return result

        # Find all valid detection indices
        valid_indices = [i for i, d in enumerate(detections) if d is not None]
        if len(valid_indices) < 2:
            return result

        # Walk consecutive pairs of valid detections and only fill the gap
        # if it's short enough to be believable.
        for a, b in zip(valid_indices, valid_indices[1:]):
            gap = b - a
            if gap <= 1 or gap > _MAX_INTERPOLATION_GAP:
                continue

            det_a = detections[a]
            det_b = detections[b]
            for i in range(a + 1, b):
                t = (i - a) / gap
                cx = det_a.center_x + t * (det_b.center_x - det_a.center_x)
                cy = det_a.center_y + t * (det_b.center_y - det_a.center_y)
                r = max(9.0, det_a.radius + t * (det_b.radius - det_a.radius))
                result[i] = BallDetection(
                    frame=i,
                    x1=cx - r,
                    y1=cy - r,
                    x2=cx + r,
                    y2=cy + r,
                    confidence=0.0,
                    interpolated=True,
                )

        return result

    def _filter_by_player_context(
        self,
        detections: list[BallDetection | None],
        player_centers_by_frame: dict[int, list[tuple[float, float]]] | None,
    ) -> list[BallDetection | None]:
        """Reject detections that are far from players unless strongly supported.

        This avoids drawing a phantom ball when the real ball goes out of frame
        and YOLO locks on to an unrelated object.
        """
        if not player_centers_by_frame:
            return detections

        result = list(detections)
        last_supported_idx = -1

        for i, det in enumerate(result):
            if det is None:
                continue

            centers = player_centers_by_frame.get(i, [])
            if centers:
                min_player_dist = min(
                    float(np.hypot(det.center_x - px, det.center_y - py))
                    for px, py in centers
                )
            else:
                min_player_dist = float("inf")

            close_to_player = min_player_dist <= _MAX_PLAYER_ASSOC_DIST
            if close_to_player:
                last_supported_idx = i
                continue

            # If not near any player, keep only if confidence is high and
            # motion remains smooth relative to the last supported detection.
            strong_confidence = det.confidence >= _MIN_UNSUPPORTED_CONF
            smooth_from_last = False
            if last_supported_idx >= 0 and result[last_supported_idx] is not None:
                prev = result[last_supported_idx]
                gap = i - last_supported_idx
                motion = float(np.hypot(det.center_x - prev.center_x, det.center_y - prev.center_y))
                smooth_from_last = motion <= (_MAX_PX_PER_FRAME * gap)

            if not (strong_confidence and smooth_from_last):
                result[i] = None
            else:
                last_supported_idx = i

        return result
