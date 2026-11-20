import pytest
from dataset_iterator import DatasetIterator


def test_iter_positive(sample_annotation_csv):
    it = DatasetIterator("zebra", annotation_path=str(sample_annotation_csv))
    assert list(it) == ["r1", "r2", "r4"]


def test_iter_empty_class_raises_stop(sample_annotation_csv):
    it = DatasetIterator("missing", annotation_path=str(sample_annotation_csv))
    with pytest.raises(StopIteration):
        next(it)


def test_iter_loop_mode(sample_annotation_csv):
    it = DatasetIterator("zebra", annotation_path=str(sample_annotation_csv))
    it.set_loop(True)
    assert [next(it) for _ in range(5)] == ["r1", "r2", "r4", "r1", "r2"]


def test_iter_reset(sample_annotation_csv):
    it = DatasetIterator("zebra", annotation_path=str(sample_annotation_csv))
    next(it)
    next(it)
    it.reset()
    assert next(it) == "r1"


def test_iter_set_loop_reset(sample_annotation_csv):
    it = DatasetIterator("zebra", annotation_path=str(sample_annotation_csv))
    next(it)
    it.set_loop(False, reset=True)
    assert next(it) == "r1"


def test_missing_annotation_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetIterator("zebra", annotation_path=str(tmp_path / "nope.csv"))