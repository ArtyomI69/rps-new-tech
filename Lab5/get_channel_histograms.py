import cv2
import numpy as np


def get_channel_histograms(df, class_label, bins=256, seed=None):
    """
    Случайно выбирает изображение заданного класса из DataFrame
    и строит гистограммы распределения яркостей по каналам B, G, R
    средствами OpenCV (cv2.calcHist).

    Параметры
    ----------
    df : pandas.DataFrame
        DataFrame с колонками 'class_label' и 'absolute_path'.
    class_label : str
        Метка класса ('bay horse' / 'zebra').
    bins : int
        Количество корзин гистограммы (по умолчанию 256).
    seed : int | None
        Seed для воспроизводимого случайного выбора.

    Возвращает
    ----------
    hist_b, hist_g, hist_r : np.ndarray
        Массивы значений гистограмм по каналам B, G, R
        (каждый длины `bins`), нормированные на количество пикселей.
    """
    # Отбираем строки нужного класса
    subset = df[df['class_label'] == class_label]
    if subset.empty:
        raise ValueError(f"В DataFrame нет изображений класса '{class_label}'")

    # Случайный выбор изображения
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subset))
    path = subset.iloc[idx]['absolute_path']

    # Чтение изображения (OpenCV: каналы BGR)
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Не удалось прочитать изображение: {path}")

    # cv2.calcHist принимает список каналов, маску, размеры и диапазоны
    hist_b = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256]).flatten()
    hist_r = cv2.calcHist([img], [2], None, [bins], [0, 256]).flatten()

    # Нормировка на общее количество пикселей (площадь под графиком = 1)
    total = img.shape[0] * img.shape[1]
    hist_b /= total
    hist_g /= total
    hist_r /= total

    print(f"Выбрано изображение: {path}")
    print(f"Размер: {img.shape[1]}×{img.shape[0]}, каналов: {img.shape[2]}")

    return hist_b, hist_g, hist_r