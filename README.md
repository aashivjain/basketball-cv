# Basketball CV

Early-stage basketball computer vision project for processing NBA game footage, tracking players and the ball, and turning those tracks into analysis-ready basketball data.

## Current Status

The project currently supports a first tracking spike with Ultralytics YOLO and a multi-object tracker. The next milestones are:

1. Detect and track players reliably. ✅
2. Export frame-level tracks to structured data. ✅
3. Add basketball detection and tracking.
4. Map image coordinates to court coordinates.
5. Derive basketball events and stats from tracks.
6. Build analysis notebooks and visualizations.

## Project Layout

```text
basketball-cv/
  configs/                 Runtime configuration files
  data/
    raw_videos/            Local source footage, not committed
  notebooks/               Exploratory analysis
  outputs/
    player_tracks.csv      Generated tracking analytics data
  src/
    basketball_cv/
      detection/           Model loading and object detection
      tracking/            Player/ball tracking pipelines
      stats/               Stat and event derivation
      io/                  Video and file I/O helpers
      pipelines/           CLI entry points
    track_players.py       Backward-compatible script entry point
  tests/                   Unit and integration tests
```

## Quick Start

### 1. Install dependencies

Create a virtual environment and install runtime requirements:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare your video

Place an NBA court video at:

```text
data/raw_videos/sample.mp4
```

**Video requirements:**
- Single camera angle (no cuts or multi-angle footage)
- Minimum 30 seconds, typically 1-5 minutes for testing
- Format: MP4, MOV, or AVI

### 3. Run the tracking pipeline

```powershell
python -m basketball_cv --config configs/player_tracking.yaml
```

**What happens:**
1. YOLO detects players in each frame
2. Multi-object tracker assigns track IDs across frames
3. Results saved to `outputs/player_tracks.csv`
4. Tracked video saved to `runs/detect/outputs/track_players-N/sample.avi`

### Output Files

**`outputs/player_tracks.csv`** — Analytics-ready tracking data

Columns:
- `frame`: Video frame number
- `track_id`: Unique player identifier (persistent across frames)
- `class_id`: Object class (0 = person)
- `confidence`: Detection confidence (0–1)
- `x1, y1, x2, y2`: Bounding box coordinates (top-left and bottom-right)
- `width, height`: Box dimensions
- `center_x, center_y`: Center point (useful for trajectory analysis)

Example rows:
```
frame,track_id,class_id,confidence,x1,y1,x2,y2,width,height,center_x,center_y
0,1,0,0.614,301.83,315.54,366.18,476.54,64.35,161.00,334.01,396.04
0,2,0,0.386,126.63,383.36,197.21,486.76,70.58,103.40,161.92,435.06
```

**`runs/detect/outputs/track_players-N/sample.avi`** — Tracked video

Visual validation of tracking:
- Bounding boxes around each detected player
- Track IDs labeled on boxes
- Same frame count as source video

## Tracking Configuration

### Occlusion & ID Reassignment Improvements

The project implements **post-processing improvements** to reduce ID reassignment when players are occluded:

**Track ID Consistency (`improve_track_consistency`)**
- **IoU-based matching (IoU > 0.3)**: Matches untracked detections to previous track IDs using spatial overlap
- **Gap bridging (max 2 frames)**: Reconnects temporary disappearances to maintain player identity through brief occlusions
- **Appearance matching**: Prioritizes spatial consistency to handle crowded moments

How it works:
1. Extract raw YOLO tracking results (may have gaps and ID resets)
2. Post-process to find untracked detections in current frame
3. Look back up to 2 frames and find best spatial match (highest IoU)
4. Assign matched detections to previous track IDs
5. Preserve remaining detections as new tracks

This reduces ID reassignment when:
- Players go behind teammates (brief 1-2 frame occlusion)
- YOLO fails to detect a player momentarily
- Crowded scenes create ambiguous tracking

To adjust behavior, modify parameters in [src/basketball_cv/tracking/players.py](src/basketball_cv/tracking/players.py#L48):
```python
improve_track_consistency(track_points, iou_threshold=0.3, max_gap_frames=2)
```

### Development

Install dev dependencies:

```powershell
pip install -r requirements-dev.txt
pytest
```

Run tests:

```powershell
pytest -v
```


