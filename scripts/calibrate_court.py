"""Interactive court calibration tool.

Opens the first frame of a video and lets you click court landmarks.
Prints the pixel coordinates so you can add them to your config YAML.

Usage:
    python scripts/calibrate_court.py data/raw_videos/sample.mp4

Click on court landmarks in order. Press 'q' to quit, 'u' to undo last point.
"""

import sys
import cv2
import numpy as np

clicked_points: list[tuple[int, int]] = []

# Known court landmarks (feet) matching the click order.
# Adjust this list to match which landmarks you plan to click.
COURT_LANDMARKS_FT = [
    ("Near-side paint baseline corner", [0, 17]),
    ("Far-side paint baseline corner", [0, 33]),
    ("Near-side free-throw line corner", [19, 17]),
    ("Far-side free-throw line corner", [19, 33]),
    ("3pt line at near baseline", [0, 3]),
    ("3pt line at far baseline", [0, 47]),
    ("Half-court near sideline", [47, 8]),
    ("Half-court far sideline", [47, 42]),
]


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"  Point {len(clicked_points)}: [{x}, {y}]")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/calibrate_court.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read first frame.")
        sys.exit(1)

    window_name = "Court Calibration - Click landmarks, Q=quit, U=undo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=== Court Calibration ===")
    print("Click the following landmarks in order:")
    for i, (name, coords) in enumerate(COURT_LANDMARKS_FT):
        print(f"  {i+1}. {name} -> court ({coords[0]}ft, {coords[1]}ft)")
    print("\nPress 'q' when done, 'u' to undo last point.\n")

    while True:
        display = frame.copy()

        # Draw clicked points
        for i, (px, py) in enumerate(clicked_points):
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.circle(display, (px, py), 6, (255, 255, 255), 2)
            label = f"{i+1}"
            cv2.putText(display, label, (px + 10, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show which point to click next
        if len(clicked_points) < len(COURT_LANDMARKS_FT):
            next_name = COURT_LANDMARKS_FT[len(clicked_points)][0]
            cv2.putText(display, f"Click: {next_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(display, "All points clicked! Press Q to finish.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('u') and clicked_points:
            removed = clicked_points.pop()
            print(f"  Undid point: [{removed[0]}, {removed[1]}]")

    cv2.destroyAllWindows()

    # Print YAML output
    if len(clicked_points) >= 4:
        print("\n\n=== Add this to your config YAML ===\n")
        print("court_calibration:")
        print("  video_points:")
        for i, (px, py) in enumerate(clicked_points):
            name = COURT_LANDMARKS_FT[i][0] if i < len(COURT_LANDMARKS_FT) else "custom"
            print(f"    - [{px}, {py}]   # {name}")
        print("  court_points:")
        for i, (px, py) in enumerate(clicked_points):
            if i < len(COURT_LANDMARKS_FT):
                coords = COURT_LANDMARKS_FT[i][1]
                name = COURT_LANDMARKS_FT[i][0]
                print(f"    - [{coords[0]}, {coords[1]}]     # {name}")
    else:
        print(f"\nOnly {len(clicked_points)} points clicked. Need at least 4 for homography.")


if __name__ == "__main__":
    main()
