"""Ball detection and tracking for basketball video analysis.

Uses YOLO (COCO class 32 = sports ball) for per-frame detection, then applies:
  1. Per-frame: best-confidence detection only (one ball in play)
  2. Motion filter: drop implausible jumps between frames
  3. Linear interpolation: fill short gaps between valid detections

Inspired by: https://github.com/abdullahtarek/basketball_analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO class index for "sports ball"
_BALL_CLASS_ID = 32

# Max pixels the ball centre can move per frame gap before the detection
# is considered a false positive.  Basketball broadcast footage is typically
# 1280×720; a fast pass covers ~30 px/frame at 30 fps.
_MAX_PX_PER_FRAME = 40


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
    """Detect and track the basketball across all video frames.

    Pipeline
    --------
    1. YOLO predict (class 32 = sports ball) streaming over the video.
    2. Keep the single highest-confidence detection per frame.
    3. Motion filter: remove detections that jump farther than
       ``_MAX_PX_PER_FRAME * gap_frames`` pixels.
    4. Linear interpolation between surviving detections.

    Usage
    -----
    >>> tracker = BallTracker(confidence=0.1)
    >>> positions = tracker.track(model, "video.mp4")
    """

    def __init__(self, confidence: float = 0.1) -> None:
        # Low threshold because COCO-trained models underestimate confidence
        # for small basketballs.  Motion filtering removes false positives.
        self.confidence = confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(
        self, model: object, video_path: str | Path
    ) -> list[BallDetection | None]:
        """Run full ball tracking pipeline on *video_path*.

        Returns a list indexed by frame number.  ``None`` means no ball
        position is available for that frame (beyond the range of
        detections, or detection failed everywhere).
        """
        raw = self._detect(model, video_path)
        filtered = self._filter_by_motion(raw)
        return self._interpolate(filtered)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _detect(
        self, model: object, video_path: str | Path
    ) -> list[BallDetection | None]:
        """Run YOLO and keep the best detection per frame."""
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
        """Drop detections that are physically implausible given prior position."""
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
        """Fill gaps with linear interpolation (reference repo technique)."""
        n = len(detections)
        if n == 0:
            return detections

        xs = np.array([d.center_x if d else np.nan for d in detections])
        ys = np.array([d.center_y if d else np.nan for d in detections])
        rs = np.array([d.radius if d else np.nan for d in detections])

        valid_mask = ~np.isnan(xs)
        if valid_mask.sum() < 2:
            return detections

        frames = np.arange(n)
        valid_frames = frames[valid_mask]
        first_valid = int(valid_frames[0])
        last_valid = int(valid_frames[-1])

        # Interpolate only within the range covered by real detections
        xs_i = np.interp(frames, valid_frames, xs[valid_mask])
        ys_i = np.interp(frames, valid_frames, ys[valid_mask])
        rs_i = np.interp(frames, valid_frames, rs[valid_mask])

        result = list(detections)
        for i in range(first_valid, last_valid + 1):
            if detections[i] is None:
                r = max(9.0, float(rs_i[i]))
                cx, cy = float(xs_i[i]), float(ys_i[i])
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
