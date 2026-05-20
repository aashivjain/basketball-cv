from pathlib import Path

from basketball_cv.tracking.track_data import TrackPoint, extract_track_points, save_track_points


class FakeBox:
    def __init__(self, xyxy, cls, conf, track_id=None):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf
        self.id = track_id


class FakeResult:
    def __init__(self, boxes, frame=None):
        self.boxes = boxes
        self.frame = frame


def test_extract_track_points_from_fake_results():
    boxes = [FakeBox([10.0, 20.0, 30.0, 40.0], cls=0, conf=0.85, track_id=1)]
    results = [FakeResult(boxes, frame=5)]

    points = extract_track_points(results)

    assert len(points) == 1
    assert points[0].frame == 5
    assert points[0].track_id == 1
    assert points[0].class_id == 0
    assert points[0].confidence == 0.85
    assert points[0].x1 == 10.0
    assert points[0].y1 == 20.0
    assert points[0].x2 == 30.0
    assert points[0].y2 == 40.0


def test_save_track_points_creates_csv(tmp_path: Path):
    points = [TrackPoint(frame=1, track_id=1, class_id=0, confidence=0.9, x1=10.0, y1=20.0, x2=30.0, y2=40.0)]
    output_file = tmp_path / "player_tracks.csv"

    saved_path = save_track_points(points, output_file)

    assert saved_path.exists()
    content = saved_path.read_text(encoding="utf-8")
    assert "frame,track_id,class_id,confidence,x1,y1,x2,y2,width,height,center_x,center_y" in content
    assert "1,1,0,0.9,10.0,20.0,30.0,40.0,20.0,20.0,20.0,30.0" in content
