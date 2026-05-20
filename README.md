# Basketball CV

Early-stage basketball computer vision project for processing NBA game footage, tracking players and the ball, and turning those tracks into analysis-ready basketball data.

## Current Status

The project currently supports a first tracking spike with Ultralytics YOLO and a multi-object tracker. The next milestones are:

1. Detect and track players reliably.
2. Add basketball detection and tracking.
3. Export frame-level tracks to structured data.
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
    tracks/                Generated track data, not committed
    videos/                Rendered/debug videos, not committed
  src/
    basketball_cv/
      detection/           Model loading and object detection
      tracking/            Player/ball tracking pipelines
      stats/               Stat and event derivation
      io/                  Video and file I/O helpers
    track_players.py       Backward-compatible script entry point
  tests/                   Lightweight tests as the project grows
```

## Quick Start

Install runtime requirements in your virtual environment:

```powershell
pip install -r requirements.txt
```

For development and tests:

```powershell
pip install -r requirements-dev.txt
```

Place a local video at:

```text
data/raw_videos/sample.mp4
```

Run player tracking using the package entry point:

```powershell
python -m basketball_cv --config configs/player_tracking.yaml
```

The pipeline now produces structured analytics-ready track data in CSV form, for example:

```text
outputs/player_tracks.csv
```

Or use the script entry point directly:

```powershell
python src/track_players.py
```

The package is designed so future work can add structured outputs, court transforms, and event analytics without changing the command-line interface.

