import pandas as pd


class DatasetIterator:
    def __init__(self, class_label, annotation_path="./annotation.csv"):
        self.class_label = class_label
        self.loop = False
        df = pd.read_csv(annotation_path, sep=";", encoding="utf-8")
        self._paths = df.loc[df["class_label"] == class_label, "relative_path"].tolist()
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._paths:
            raise StopIteration
        if self._index >= len(self._paths):
            if self.loop:
                self._index = 0
            else:
                raise StopIteration
        path = self._paths[self._index]
        self._index += 1
        return path

    def set_loop(self, loop: bool, reset: bool = False) -> None:
        """
        Включает или выключает режим зацикливания.

        :param loop: True — итератор будет бесконечно ходить по кругу,
                     False — остановится (StopIteration) после последнего элемента.
        :param reset: если True — сбрасывает текущую позицию в начало.
        """
        self.loop = bool(loop)
        if reset:
            self._index = 0

    def reset(self) -> None:
        """Сбрасывает итератор в начало."""
        self._index = 0