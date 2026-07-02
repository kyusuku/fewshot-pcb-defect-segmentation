from datasets.deeppcb import parse_deeppcb_annotation_line
from datasets.sampling import sample_few_shot_normals
from datasets.types import DatasetRecord


def test_parse_deeppcb_annotation_line():
    box = parse_deeppcb_annotation_line("10,20,30,40,3")

    assert box.to_xyxy() == (10.0, 20.0, 30.0, 40.0)
    assert box.class_id == 3
    assert box.class_name == "mousebite"


def test_sample_few_shot_normals_is_category_balanced(tmp_path):
    records = [
        DatasetRecord("visa_pcb", "pcb1", "pcb1/a", "train", tmp_path / "a.jpg", 0),
        DatasetRecord("visa_pcb", "pcb1", "pcb1/b", "train", tmp_path / "b.jpg", 0),
        DatasetRecord("visa_pcb", "pcb1", "pcb1/bad", "test", tmp_path / "bad.jpg", 1),
        DatasetRecord("visa_pcb", "pcb2", "pcb2/a", "train", tmp_path / "c.jpg", 0),
        DatasetRecord("visa_pcb", "pcb2", "pcb2/b", "train", tmp_path / "d.jpg", 0),
    ]

    support = sample_few_shot_normals(records, k=1, seed=4880)

    assert set(support) == {"pcb1", "pcb2"}
    assert all(len(samples) == 1 for samples in support.values())
    assert all(sample.label == 0 for samples in support.values() for sample in samples)
