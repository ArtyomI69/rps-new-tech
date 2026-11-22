import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------- Загрузка изображений ----------
img1 = cv2.imread('img1.png')            # 3 яблока
img2 = cv2.imread('img2.png')            # цветок крупным планом
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

def show(images, titles, figsize=(16, 5), gray_cmap='gray'):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1: axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap=gray_cmap)
        else:
            ax.imshow(im)
        ax.set_title(t, fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ---------- 2.1 Предобработка ----------
gray  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (7, 7), 0)          # сглаживание шума

# Бинаризация методом Оцу + инверсия (яблоки темнее фона)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Морфологическая чистка
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
clean  = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel, iterations=1)

# ---------- 2.2 Canny как альтернатива ----------
edges = cv2.Canny(blur, 50, 150)

# ---------- 2.3 Поиск и фильтрация контуров ----------
contours, hierarchy = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
# Оставляем только крупные объекты (> 0.5% площади)
area_img = gray.shape[0] * gray.shape[1]
big = [c for c in contours if cv2.contourArea(c) > 0.005 * area_img]

vis = img1.copy()
for i, c in enumerate(big):
    cv2.drawContours(vis, [c], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    M = cv2.moments(c)
    if M['m00'] > 0:
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis, f'#{i}', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

show([cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), blur, thresh, clean,
      edges, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)],
     ['Оригинал', 'Gaussian blur (7×7)', 'Otsu (инвертир.)',
      'Морфологич. чистка', 'Canny (50,150)', 'Контуры + bounding box'],
     figsize=(18, 4))
print(f'Найдено контуров после фильтрации: {len(big)}')