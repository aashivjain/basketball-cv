"""Video renderer with tracking annotations and full NBA court overlay."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from basketball_cv.tracking.track_data import TrackPoint

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


def _draw_full_court(width: int, height: int) -> np.ndarray:
    """Draw a full NBA court (94ft x 50ft) top-down view.

    Left basket on the left, right basket on the right.
    Includes: boundary, half-court line, center circle, both three-point arcs,
    both paints/keys, free-throw lines/circles, restricted areas, rims.
    """
    court = np.zeros((height, width, 3), dtype=np.uint8)
    # Hardwood floor color
    court[:] = (40, 65, 115)

    mx, my = 8, 6
    cw = width - 2 * mx   # usable drawing width
    ch = height - 2 * my  # usable drawing height

    # Full court: 94ft long (x-axis) x 50ft wide (y-axis)
    sx = cw / 94.0  # pixels per foot along length
    sy = ch / 50.0  # pixels per foot along width

    white = (220, 220, 220)
    gray = (130, 130, 130)
    orange = (0, 120, 255)

    def fp(x_ft: float, y_ft: float) -> tuple[int, int]:
        """Court feet → pixel coords. (0,0) = left-bottom corner."""
        return (mx + int(x_ft * sx), my + ch - int(y_ft * sy))

    # --- Outer boundary ---
    cv2.rectangle(court, fp(0, 0), fp(94, 50), white, 2)

    # --- Half-court line ---
    cv2.line(court, fp(47, 0), fp(47, 50), white, 1)

    # --- Center circle (6ft radius) ---
    cc = fp(47, 25)
    cv2.circle(court, cc, int(6 * sx), white, 1, cv2.LINE_AA)

    # --- Left side (basket at x=5.25, y=25) ---
    # Paint: 16ft wide (y: 17-33), 19ft from baseline (x: 0-19)
    cv2.rectangle(court, fp(0, 17), fp(19, 33), white, 1)
    # Free-throw line
    cv2.line(court, fp(19, 17), fp(19, 33), white, 1)
    # Free-throw circle
    ft_l = fp(19, 25)
    cv2.ellipse(court, ft_l, (int(6 * sy), int(6 * sx)), 90, 0, 180, white, 1, cv2.LINE_AA)
    cv2.ellipse(court, ft_l, (int(6 * sy), int(6 * sx)), 90, 180, 360, gray, 1, cv2.LINE_AA)
    # Restricted area (4ft radius)
    basket_l = fp(5.25, 25)
    cv2.ellipse(court, basket_l, (int(4 * sy), int(4 * sx)), 90, 0, 180, white, 1, cv2.LINE_AA)
    # Three-point line: straight at y=3 and y=47 from baseline to ~14ft
    cv2.line(court, fp(0, 3), fp(14, 3), white, 1)
    cv2.line(court, fp(0, 47), fp(14, 47), white, 1)
    # Three-point arc (23.75ft radius from basket)
    arc_l = fp(5.25, 25)
    cv2.ellipse(court, arc_l, (int(23.75 * sx), int(23.75 * sy)), 0, -73, 73, white, 1, cv2.LINE_AA)
    # Backboard
    cv2.line(court, fp(4, 22), fp(4, 28), (180, 180, 200), 2)
    # Rim
    cv2.circle(court, basket_l, max(2, int(0.75 * sy)), orange, 1, cv2.LINE_AA)

    # --- Right side (basket at x=88.75, y=25) ---
    # Paint
    cv2.rectangle(court, fp(75, 17), fp(94, 33), white, 1)
    # Free-throw line
    cv2.line(court, fp(75, 17), fp(75, 33), white, 1)
    # Free-throw circle
    ft_r = fp(75, 25)
    cv2.ellipse(court, ft_r, (int(6 * sy), int(6 * sx)), 90, 180, 360, white, 1, cv2.LINE_AA)
    cv2.ellipse(court, ft_r, (int(6 * sy), int(6 * sx)), 90, 0, 180, gray, 1, cv2.LINE_AA)
    # Restricted area
    basket_r = fp(88.75, 25)
    cv2.ellipse(court, basket_r, (int(4 * sy), int(4 * sx)), 90, 180, 360, white, 1, cv2.LINE_AA)
    # Three-point line straight portions
    cv2.line(court, fp(80, 3), fp(94, 3), white, 1)
    cv2.line(court, fp(80, 47), fp(94, 47), white, 1)
    # Three-point arc
    arc_r = fp(88.75, 25)
    cv2.ellipse(court, arc_r, (int(23.75 * sx), int(23.75 * sy)), 0, 107, 253, white, 1, cv2.LINE_AA)
    # Backboard
    cv2.line(court, fp(90, 22), fp(90, 28), (180, 180, 200), 2)
    # Rim
    cv2.circle(court, basket_r, max(2, int(0.75 * sy)), orange, 1, cv2.LINE_AA)

    return court


def render_tracked_video(
    input_video: str | Path,
    output_path: str | Path,
    track_points: list[TrackPoint],
) -> Path:
    """Render video with bounding boxes and full-court mini-map.

    - Color-coded bounding boxes with ID labels
    - Full NBA court overlay (top-right)
    - Foot position (bottom-center of bbox) mapped to court
    - EMA-smoothed positions for natural movement
    - Per-player fading trails
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

    # Index track points by frame (skip None IDs)
    frame_tracks: dict[int, list[TrackPoint]] = {}
    for pt in track_points:
        if pt.track_id is None:
            continue
        frame_tracks.setdefault(pt.frame, []).append(pt)

    # Full-court overlay (wider aspect ratio: 94:50 ≈ 1.88:1)
    court_w, court_h = 280, 150
    base_court = _draw_full_court(court_w, court_h)
    court_mx, court_my = 8, 6
    court_cw = court_w - 2 * court_mx
    court_ch = court_h - 2 * court_my

    # Smoothing state
    smooth: dict[int, tuple[float, float]] = {}
    trails: dict[int, deque[tuple[float, float]]] = {}
    ema_alpha = 0.4  # smoothing factor

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = frame_tracks.get(frame_idx, [])

        # --- Bounding boxes ---
        for pt in detections:
            color = _color_for(pt.track_id)
            x1, y1, x2, y2 = int(pt.x1), int(pt.y1), int(pt.x2), int(pt.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"#{pt.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # --- Court overlay ---
        court_img = base_court.copy()

        for pt in detections:
            tid = pt.track_id
            # Foot position (bottom-center)
            foot_x = (pt.x1 + pt.x2) / 2.0
            foot_y = pt.y2

            # Normalize to [0, 1]
            nx = foot_x / vid_w
            ny = foot_y / vid_h

            # EMA smooth
            if tid in smooth:
                px, py = smooth[tid]
                nx = ema_alpha * nx + (1 - ema_alpha) * px
                ny = ema_alpha * ny + (1 - ema_alpha) * py
            smooth[tid] = (nx, ny)

            # Append to trail
            if tid not in trails:
                trails[tid] = deque(maxlen=60)
            trails[tid].append((nx, ny))

        # Draw trails
        for tid, trail in trails.items():
            if len(trail) < 2:
                continue
            color = _color_for(tid)
            for i in range(1, len(trail)):
                fade = i / len(trail)
                c = tuple(int(v * fade * 0.6) for v in color)
                p1 = (court_mx + int(trail[i - 1][0] * court_cw),
                       court_my + int(trail[i - 1][1] * court_ch))
                p2 = (court_mx + int(trail[i][0] * court_cw),
                       court_my + int(trail[i][1] * court_ch))
                cv2.line(court_img, p1, p2, c, 1, cv2.LINE_AA)

        # Draw current player dots
        for pt in detections:
            tid = pt.track_id
            if tid in smooth:
                nx, ny = smooth[tid]
                cx = court_mx + int(nx * court_cw)
                cy = court_my + int(ny * court_ch)
                color = _color_for(tid)
                cv2.circle(court_img, (cx, cy), 4, color, -1, cv2.LINE_AA)
                cv2.circle(court_img, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend onto frame (top-right)
        ox = vid_w - court_w - 8
        oy = 8
        roi = frame[oy:oy + court_h, ox:ox + court_w]
        if roi.shape[:2] == (court_h, court_w):
            blended = cv2.addWeighted(roi, 0.15, court_img, 0.85, 0)
            frame[oy:oy + court_h, ox:ox + court_w] = blended

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path
