import pandas as pd


class DatasetIterator:
    """
    Итератор по relative_path изображений одного класса из annotation.csv
    (разделитель ';', UTF-8). Состояние хранится в самом объекте:
    повторный for продолжит с текущей позиции, а не начнёт заново.

    По умолчанию обход одноразовый (StopIteration в конце);
    бесконечный цикл включается через set_loop(True).
    """
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
        """Включает/выключает зацикливание; reset=True — сброс в начало."""
        self.loop = bool(loop)
        if reset:
            self._index = 0

    def reset(self) -> None:
        """Сбрасывает итератор в начало."""
        self._index = 0