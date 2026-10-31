import cv2

def get_image_info(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    height, width = img.shape[:2]
    depth = 1 if img.ndim == 2 else img.shape[2]
    return height, width, depth