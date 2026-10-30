# dataset_iterator.py
import pandas as pd


class DatasetIterator:
    """Итератор по изображениям одного класса из CSV-аннотации."""

    def __init__(self, class_label, annotation_path, project_root=None):
        self.class_label = class_label
        self.annotation_path = annotation_path
        self.project_root = project_root

        df = pd.read_csv(annotation_path, sep=";", encoding="utf-8-sig")
        self._paths = df.loc[
            df["class_label"] == class_label, "relative_path"
        ].tolist()
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._paths):
            raise StopIteration
        path = self._paths[self._index]
        self._index += 1
        return path

    def reset(self):
        self._index = 0

    def __len__(self):
        return len(self._paths)


def get_classes(annotation_path):
    """Возвращает отсортированный список классов из аннотации."""
    df = pd.read_csv(annotation_path, sep=";", encoding="utf-8-sig")
    return sorted(df["class_label"].dropna().unique().tolist())