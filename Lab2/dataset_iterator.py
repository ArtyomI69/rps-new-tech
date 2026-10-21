import pandas as pd

class DatasetIterator:
    def __init__(self, class_label, annotation_path="./annotation.csv"):
        self.class_label = class_label
        df = pd.read_csv(annotation_path, sep=";", encoding="utf-8")
        self._paths = df.loc[df["class_label"] == class_label, "relative_path"].tolist()
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._paths):
            raise StopIteration
        path = self._paths[self._index]
        self._index += 1
        return path