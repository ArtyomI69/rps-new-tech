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

h, w = img1.shape[:2]
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# ---------- 5.1 Поворот ----------
M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle=30, scale=1.0)
rotated = cv2.warpAffine(img, M_rot, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

# ---------- 5.2 Масштабирование ----------
scaled_up   = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
scaled_down = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# ---------- 5.3 Сдвиг ----------
M_shift = np.float32([[1, 0, 60], [0, 1, 40]])
shifted = cv2.warpAffine(img, M_shift, (w, h))

# ---------- 5.4 Яркость и контраст ----------
bright   = cv2.convertScaleAbs(img, alpha=1.0, beta=50)    # ярче
dark     = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)   # темнее
contrast = cv2.convertScaleAbs(img, alpha=1.6, beta=0)     # контраст
gamma    = np.power(img/255.0, 1/2.2) * 255
gamma    = gamma.astype(np.uint8)                          # гамма-коррекция

# ---------- 5.5 Фильтрация ----------
gauss   = cv2.GaussianBlur(img, (7, 7), 0)
median  = cv2.medianBlur(img, 7)
sobel_x = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                    cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
lap     = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
lap     = cv2.convertScaleAbs(lap)

show([img, rotated, scaled_up, scaled_down, shifted],
     ['Оригинал', 'Поворот 30°', 'Масштаб ×1.5 (CUBIC)',
      'Масштаб ×0.5 (AREA)', 'Сдвиг (+60,+40)'], figsize=(18, 4))

show([img, bright, dark, contrast, gamma],
     ['Оригинал', 'Ярче (β=+50)', 'Темнее (β=−50)',
      'Контраст (α=1.6)', 'Гамма 1/2.2'], figsize=(18, 4))

show([img, gauss, median, sobel_x, lap],
     ['Оригинал', 'Gaussian 7×7', 'Median 7×7',
      'Sobel X (края)', 'Laplacian (края)'], figsize=(18, 4))