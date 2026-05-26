"""Per-frame court detection and coordinate mapping.

Detects the basketball court paint (key) via color segmentation in each frame,
then uses the paint's 4 corners as anchor points to compute a homography
mapping player positions to 2D court coordinates.

This approach is robust to camera panning since it detects the court
in every frame independently.
"""

from __future__ import annotations

import numpy as np
import cv2


# NBA court dimensions in feet
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0

# Paint (key) dimensions
PAINT_LENGTH_FT = 19.0  # from baseline to free-throw line
PAINT_NEAR_Y = 17.0     # near-side paint edge (y-coordinate)
PAINT_FAR_Y = 33.0      # far-side paint edge (y-coordinate)

# Court coordinates for the 4 corners of each paint:
# Order: TL (far-baseline), TR (far-FT), BR (near-FT), BL (near-baseline)
LEFT_PAINT_COURT = np.array([
    [0, PAINT_FAR_Y],       # far-side baseline corner
    [PAINT_LENGTH_FT, PAINT_FAR_Y],  # far-side FT line corner
    [PAINT_LENGTH_FT, PAINT_NEAR_Y], # near-side FT line corner
    [0, PAINT_NEAR_Y],      # near-side baseline corner
], dtype=np.float32)

RIGHT_PAINT_COURT = np.array([
    [COURT_LENGTH_FT - PAINT_LENGTH_FT, PAINT_FAR_Y],  # far-side FT line corner
    [COURT_LENGTH_FT, PAINT_FAR_Y],    # far-side baseline corner
    [COURT_LENGTH_FT, PAINT_NEAR_Y],   # near-side baseline corner
    [COURT_LENGTH_FT - PAINT_LENGTH_FT, PAINT_NEAR_Y], # near-side FT line corner
], dtype=np.float32)


class CourtMapper:
    """Per-frame court mapper using paint detection.

    Detects the purple paint rectangle in each frame via HSV color segmentation,
    then computes a homography from the detected corners to known court coordinates.
    """

    def __init__(
        self,
        court_width_px: int = 300,
        court_height_px: int = 161,
    ) -> None:
        self.court_width_px = court_width_px
        self.court_height_px = court_height_px
        self._mx = 8
        self._my = 6
        self._current_matrix: np.ndarray | None = None
        self._last_good_matrix: np.ndarray | None = None

    def _feet_to_overlay_px(self, x_ft: float, y_ft: float) -> list[float]:
        """Convert court feet to overlay pixel coordinates."""
        cw = self.court_width_px - 2 * self._mx
        ch = self.court_height_px - 2 * self._my
        px_x = self._mx + (x_ft / COURT_LENGTH_FT) * cw
        px_y = self._my + (1.0 - y_ft / COURT_WIDTH_FT) * ch
        return [px_x, px_y]

    def detect_and_update(self, frame: np.ndarray) -> bool:
        """Detect the paint in the current frame and update the homography.

        Returns True if detection succeeded and mapping was updated.
        """
        corners, is_left = _detect_paint_corners(frame)
        if corners is None:
            # Use last good mapping if available
            if self._last_good_matrix is not None:
                self._current_matrix = self._last_good_matrix
            return False

        # Map detected corners to known court coordinates
        if is_left:
            court_feet = LEFT_PAINT_COURT
        else:
            court_feet = RIGHT_PAINT_COURT

        # Convert court feet to overlay pixel coordinates
        court_overlay = np.array([
            self._feet_to_overlay_px(pt[0], pt[1])
            for pt in court_feet
        ], dtype=np.float32)

        # Compute homography from detected pixel corners → court overlay pixels
        # Use direct least-squares (fast) since paint corners are clean and reliable
        H, _ = cv2.findHomography(corners, court_overlay)
        if H is None:
            if self._last_good_matrix is not None:
                self._current_matrix = self._last_good_matrix
            return False

        self._current_matrix = H
        self._last_good_matrix = H.copy()
        return True

    def transform_foot_position(self, foot_x: float, foot_y: float) -> tuple[float, float] | None:
        """Transform a player's foot position to court overlay pixel coordinates.

        Returns None if no valid mapping exists or position is off-court.
        """
        if self._current_matrix is None:
            return None

        pts = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self._current_matrix)
        x, y = float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

        # Check bounds with margin
        margin = 10
        if x < -margin or x > self.court_width_px + margin:
            return None
        if y < -margin or y > self.court_height_px + margin:
            return None

        # Clamp to overlay bounds
        x = max(0, min(self.court_width_px - 1, x))
        y = max(0, min(self.court_height_px - 1, y))

        return (x, y)


def _detect_paint_corners(frame: np.ndarray) -> tuple[np.ndarray | None, bool | None]:
    """Detect the basketball paint (key) rectangle via purple color segmentation.

    Returns:
        (corners, is_left): 4x2 array of ordered corners [TL, TR, BR, BL]
        and whether it's the left paint. Returns (None, None) if not detected.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Purple/violet range - tuned for Lakers court paint
    lower_purple = np.array([120, 40, 40])
    upper_purple = np.array([165, 255, 200])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Morphological cleanup (3x3 kernel: faster than 5x5, still effective)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Find the largest contour that's paint-shaped
    best_contour = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        # Check aspect ratio of bounding rect - paint should be wider than tall
        # in a broadcast side-view
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if 1.5 < aspect < 8.0 and area > best_area:
            best_area = area
            best_contour = c

    if best_contour is None:
        return None, None

    # Get the 4 corners of the paint quadrilateral
    corners = _get_quadrilateral_corners(best_contour)
    if corners is None:
        return None, None

    # Determine if left or right paint based on position in frame
    frame_center_x = frame.shape[1] / 2
    paint_center_x = corners[:, 0].mean()
    is_left = paint_center_x < frame_center_x

    return corners, is_left


def _get_quadrilateral_corners(contour: np.ndarray) -> np.ndarray | None:
    """Extract 4 ordered corners from a contour.

    Returns corners in order: TL, TR, BR, BL (relative to image coordinates).
    """
    # Try polygon approximation first
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        # Fall back to minimum area rotated rectangle
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.float32)

    # Order corners: TL (top-left), TR (top-right), BR (bottom-right), BL (bottom-left)
    # Split into top/bottom by y-coordinate
    center_y = corners[:, 1].mean()
    top_mask = corners[:, 1] < center_y
    bottom_mask = ~top_mask

    top = corners[top_mask]
    bottom = corners[bottom_mask]

    if len(top) != 2 or len(bottom) != 2:
        # Fallback: sort by y and split
        sorted_idx = corners[:, 1].argsort()
        top = corners[sorted_idx[:2]]
        bottom = corners[sorted_idx[2:]]

    # Within top/bottom, sort by x
    tl = top[top[:, 0].argmin()]
    tr = top[top[:, 0].argmax()]
    bl = bottom[bottom[:, 0].argmin()]
    br = bottom[bottom[:, 0].argmax()]

    return np.array([tl, tr, br, bl], dtype=np.float32)
