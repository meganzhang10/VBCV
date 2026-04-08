from pathlib import Path

from src.detection.dataset import ClipRecord, load_manifest, write_yolo_dataset_yaml
from src.detection.evaluate import load_yolo_label


def test_load_manifest_reads_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "file_name,source,angle,fps,resolution,status",
                "clip_a.mp4,local,behind-server,60,1920x1080,pending",
                "clip_b.mp4,local,behind-server,30,1920x1080,pending",
            ]
        ),
        encoding="utf-8",
    )
    rows = load_manifest(manifest)
    assert rows == [
        ClipRecord(
            file_name="clip_a.mp4",
            source="local",
            angle="behind-server",
            fps=60,
            resolution="1920x1080",
            status="pending",
        ),
        ClipRecord(
            file_name="clip_b.mp4",
            source="local",
            angle="behind-server",
            fps=30,
            resolution="1920x1080",
            status="pending",
        ),
    ]


def test_write_yolo_dataset_yaml(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ball_dataset"
    dataset_root.mkdir()
    out_yaml = tmp_path / "dataset.yaml"
    result_path = write_yolo_dataset_yaml(dataset_root, out_yaml)
    assert result_path == out_yaml
    text = out_yaml.read_text(encoding="utf-8")
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "0: volleyball" in text


def test_load_yolo_label_converts_to_pixels(tmp_path: Path) -> None:
    label = tmp_path / "frame_001.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    detections = load_yolo_label(label, image_width=1000, image_height=500)
    assert len(detections) == 1
    det = detections[0]
    assert det.class_id == 0
    assert det.x1 == 400.0
    assert det.y1 == 200.0
    assert det.x2 == 600.0
    assert det.y2 == 300.0
