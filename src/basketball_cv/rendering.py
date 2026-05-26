"""Renders the output video: bounding boxes, player IDs, court overlay, and ball trail."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from basketball_cv.tracking.track_data import TrackPoint
from basketball_cv.tracking.ball import BallDetection
from basketball_cv.court import CourtMapper

# Orange (BGR) for ball trail/dot
_BALL_COLOR = (0, 140, 255)
_BALL_TRAIL_LEN = 6
# Bright yellow indicator for the player currently holding the ball
_CARRIER_COLOR = (0, 220, 255)
_CARRIER_HOLD_FRAMES = 8
_CARRIER_SWITCH_CONFIRM_FRAMES = 2

# Distinct colors for track IDs (BGR)
_COLORS = [
    (0, 255, 127),   # spring green
    (255, 50, 50),   # blue
    (50, 50, 255),   # red
    (255, 255, 50),  # cyan
    (50, 255, 255),  # yellow
    (255, 50, 255),  # magenta
    (128, 255, 0),   # chartreuse
    (255, 165, 0),   # orange-blue
    (0, 200, 200),   # dark yellow
    (200, 100, 255), # pink
    (100, 255, 200), # aquamarine
    (200, 200, 100), # pale teal
]


def _color_for(track_id: int) -> tuple[int, int, int]:
    return _COLORS[track_id % len(_COLORS)]


def _find_ball_carrier(
    detections: list[TrackPoint],
    ball_det: BallDetection | None,
    max_dist_px: float = 110.0,
) -> int | None:
    """Return the track_id of the player closest to the ball.

    Returns None if no player is within *max_dist_px* of the ball center,
    which typically means the ball is in the air or being passed.
    """
    if ball_det is None or not detections:
        return None

    bx, by = ball_det.center_x, ball_det.center_y
    best_id: int | None = None
    best_dist = max_dist_px

    for pt in detections:
        px = (pt.x1 + pt.x2) / 2.0
        py = (pt.y1 + pt.y2) / 2.0
        dist = float(np.hypot(bx - px, by - py))
        if dist < best_dist:
            best_dist = dist
            best_id = pt.track_id

    return best_id


def _draw_full_court(width: int, height: int) -> np.ndarray:
    """Draw a full NBA court (94ft x 50ft) top-down view."""
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[:] = (40, 65, 115)

    mx, my = 8, 6
    cw = width - 2 * mx
    ch = height - 2 * my
    sx = cw / 94.0
    sy = ch / 50.0

    white = (220, 220, 220)
    gray = (130, 130, 130)
    orange = (0, 120, 255)

    def fp(x_ft: float, y_ft: float) -> tuple[int, int]:
        """Court feet -> pixel coords. (0,0) = left-bottom corner."""
        return (mx + int(x_ft * sx), my + ch - int(y_ft * sy))

    # Outer boundary
    cv2.rectangle(court, fp(0, 0), fp(94, 50), white, 2)
    # Half-court line
    cv2.line(court, fp(47, 0), fp(47, 50), white, 1)
    # Center circle
    cc = fp(47, 25)
    cv2.circle(court, cc, int(6 * sx), white, 1, cv2.LINE_AA)

    # --- Left side ---
    cv2.rectangle(court, fp(0, 17), fp(19, 33), white, 1)
    cv2.line(court, fp(19, 17), fp(19, 33), white, 1)
    ft_l = fp(19, 25)
    cv2.ellipse(court, ft_l, (int(6 * sy), int(6 * sx)), 90, 0, 180, white, 1, cv2.LINE_AA)
    cv2.ellipse(court, ft_l, (int(6 * sy), int(6 * sx)), 90, 180, 360, gray, 1, cv2.LINE_AA)
    basket_l = fp(5.25, 25)
    cv2.ellipse(court, basket_l, (int(4 * sy), int(4 * sx)), 90, 0, 180, white, 1, cv2.LINE_AA)
    cv2.line(court, fp(0, 3), fp(14, 3), white, 1)
    cv2.line(court, fp(0, 47), fp(14, 47), white, 1)
    cv2.ellipse(court, basket_l, (int(23.75 * sx), int(23.75 * sy)), 0, -73, 73, white, 1, cv2.LINE_AA)
    cv2.line(court, fp(4, 22), fp(4, 28), (180, 180, 200), 2)
    cv2.circle(court, basket_l, max(2, int(0.75 * sy)), orange, 1, cv2.LINE_AA)

    # --- Right side ---
    cv2.rectangle(court, fp(75, 17), fp(94, 33), white, 1)
    cv2.line(court, fp(75, 17), fp(75, 33), white, 1)
    ft_r = fp(75, 25)
    cv2.ellipse(court, ft_r, (int(6 * sy), int(6 * sx)), 90, 180, 360, white, 1, cv2.LINE_AA)
    cv2.ellipse(court, ft_r, (int(6 * sy), int(6 * sx)), 90, 0, 180, gray, 1, cv2.LINE_AA)
    basket_r = fp(88.75, 25)
    cv2.ellipse(court, basket_r, (int(4 * sy), int(4 * sx)), 90, 180, 360, white, 1, cv2.LINE_AA)
    cv2.line(court, fp(80, 3), fp(94, 3), white, 1)
    cv2.line(court, fp(80, 47), fp(94, 47), white, 1)
    cv2.ellipse(court, basket_r, (int(23.75 * sx), int(23.75 * sy)), 0, 107, 253, white, 1, cv2.LINE_AA)
    cv2.line(court, fp(90, 22), fp(90, 28), (180, 180, 200), 2)
    cv2.circle(court, basket_r, max(2, int(0.75 * sy)), orange, 1, cv2.LINE_AA)

    return court


def render_tracked_video(
    input_video: str | Path,
    output_path: str | Path,
    track_points: list[TrackPoint],
    court_mapper: CourtMapper | None = None,
    ball_detections: list[BallDetection | None] | None = None,
) -> Path:
    """Write an annotated copy of *input_video* to *output_path*.

    Draws colour-coded player bounding boxes, a court mini-map in the
    top-right corner with player dot positions, and an orange ball trail.
    """
    input_video = Path(input_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (vid_w, vid_h))

    # Index track points by frame
    frame_tracks: dict[int, list[TrackPoint]] = {}
    for pt in track_points:
        if pt.track_id is None:
            continue
        frame_tracks.setdefault(pt.frame, []).append(pt)

    # Court overlay dimensions (94:50 aspect ratio)
    court_w, court_h = 300, 161
    base_court = _draw_full_court(court_w, court_h)

    # Ball trail: deque of (cx, cy) pixel positions
    ball_trail: deque[tuple[int, int]] = deque(maxlen=_BALL_TRAIL_LEN)

    # Carrier state with hysteresis so indicator does not flicker.
    active_carrier_id: int | None = None
    carrier_missing_frames = 0
    pending_carrier_id: int | None = None
    pending_carrier_frames = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = frame_tracks.get(frame_idx, [])
        ball_det = (
            ball_detections[frame_idx]
            if ball_detections and frame_idx < len(ball_detections)
            else None
        )

        # Detect court paint and update mapping for this frame
        if court_mapper is not None:
            court_mapper.detect_and_update(frame)

        candidate_carrier_id = _find_ball_carrier(detections, ball_det)
        present_ids = {pt.track_id for pt in detections if pt.track_id is not None}

        if candidate_carrier_id is not None:
            carrier_missing_frames = 0
            if candidate_carrier_id == active_carrier_id:
                pending_carrier_id = None
                pending_carrier_frames = 0
            else:
                if candidate_carrier_id == pending_carrier_id:
                    pending_carrier_frames += 1
                else:
                    pending_carrier_id = candidate_carrier_id
                    pending_carrier_frames = 1

                if pending_carrier_frames >= _CARRIER_SWITCH_CONFIRM_FRAMES:
                    active_carrier_id = candidate_carrier_id
                    pending_carrier_id = None
                    pending_carrier_frames = 0
        else:
            pending_carrier_id = None
            pending_carrier_frames = 0
            carrier_missing_frames += 1
            if carrier_missing_frames > _CARRIER_HOLD_FRAMES:
                active_carrier_id = None

        # If the tracked carrier is currently out of frame, avoid showing stale UI.
        if active_carrier_id not in present_ids:
            if carrier_missing_frames > _CARRIER_HOLD_FRAMES:
                active_carrier_id = None

        carrier_id = active_carrier_id

        # --- Bounding boxes ---
        for pt in detections:
            color = _color_for(pt.track_id)
            has_ball = pt.track_id == carrier_id
            x1, y1, x2, y2 = int(pt.x1), int(pt.y1), int(pt.x2), int(pt.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if has_ball else 2)
            label = f"#{pt.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            if has_ball:
                # Downward-pointing triangle above the player's label
                cx = (x1 + x2) // 2
                tip_y = y1 - th - 8
                tri = np.array(
                    [[cx - 7, tip_y - 11], [cx + 7, tip_y - 11], [cx, tip_y]],
                    dtype=np.int32,
                )
                cv2.fillPoly(frame, [tri], _CARRIER_COLOR)
                cv2.polylines(frame, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)

        # --- Ball trail + ball dot on main video ---
        if ball_det is not None:
            ball_trail.append((int(ball_det.center_x), int(ball_det.center_y)))

        trail_list = list(ball_trail)
        for i, (tx, ty) in enumerate(trail_list):
            # i=0 is oldest; scale size and color intensity toward the current frame
            frac = (i + 1) / _BALL_TRAIL_LEN
            r = max(3, int(9 * frac))
            b_ch = int(_BALL_COLOR[0] * frac)
            g_ch = int(_BALL_COLOR[1] * frac)
            r_ch = int(_BALL_COLOR[2] * frac)
            cv2.circle(frame, (tx, ty), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), r, (b_ch, g_ch, r_ch), -1, cv2.LINE_AA)

        # Current ball: prominent circle with outline
        if ball_det is not None:
            bx, by = int(ball_det.center_x), int(ball_det.center_y)
            cv2.circle(frame, (bx, by), 11, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 10, _BALL_COLOR, -1, cv2.LINE_AA)
            if not ball_det.interpolated:
                # White highlight dot for confirmed detections
                cv2.circle(frame, (bx - 3, by - 3), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # --- Court overlay with player dots ---
        court_img = base_court.copy()

        if court_mapper is not None:
            # Homography-based mapping (proper perspective transform)
            for pt in detections:
                foot_x = (pt.x1 + pt.x2) / 2.0
                foot_y = pt.y2
                court_pos = court_mapper.transform_foot_position(foot_x, foot_y)
                if court_pos is None:
                    continue
                cx, cy = int(court_pos[0]), int(court_pos[1])
                color = _color_for(pt.track_id)
                cv2.circle(court_img, (cx, cy), 5, color, -1, cv2.LINE_AA)
                cv2.circle(court_img, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Fallback: simple normalized mapping (no perspective correction)
            for pt in detections:
                foot_x = (pt.x1 + pt.x2) / 2.0
                foot_y = pt.y2
                nx = foot_x / vid_w
                ny = foot_y / vid_h
                cx = int(nx * court_w)
                cy = int(ny * court_h)
                color = _color_for(pt.track_id)
                cv2.circle(court_img, (cx, cy), 5, color, -1, cv2.LINE_AA)
                cv2.circle(court_img, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Ball dot on court overlay
        if court_mapper is not None and ball_det is not None:
            ball_court = court_mapper.transform_foot_position(
                ball_det.center_x, ball_det.center_y
            )
            if ball_court is not None:
                bcx, bcy = int(ball_court[0]), int(ball_court[1])
                cv2.circle(court_img, (bcx, bcy), 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(court_img, (bcx, bcy), 4, _BALL_COLOR, -1, cv2.LINE_AA)

        # Blend court overlay onto frame (top-right)
        ox = vid_w - court_w - 10
        oy = 10
        roi = frame[oy:oy + court_h, ox:ox + court_w]
        if roi.shape[:2] == (court_h, court_w):
            blended = cv2.addWeighted(roi, 0.15, court_img, 0.85, 0)
            frame[oy:oy + court_h, ox:ox + court_w] = blended

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path
