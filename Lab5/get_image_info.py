import cv2

def get_image_info(path):
    """Возвращает (height, width, depth) изображения по пути.

    depth — число каналов: 1 для grayscale, 3/4 для цветных (BGR/BGRA).
    IMREAD_UNCHANGED нужен, чтобы сохранить alpha-канал и 16-бит.
    Бросает ValueError, если файл не открылся.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    height, width = img.shape[:2]
    depth = 1 if img.ndim == 2 else img.shape[2]
    return height, width, depth