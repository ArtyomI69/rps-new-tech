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

# ---------- 4.1 Бинарная маска (изображение яблок) ----------
gray1  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, binmask = cv2.threshold(gray1, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ---------- 4.2 Базовые операции ----------
erode   = cv2.erode(binmask,  k, iterations=2)
dilate  = cv2.dilate(binmask, k, iterations=2)
opening = cv2.morphologyEx(binmask, cv2.MORPH_OPEN,    k)   # erosion → dilation
closing = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE,   k)   # dilation → erosion
grad    = cv2.morphologyEx(binmask, cv2.MORPH_GRADIENT, k)  # dilate − erode
tophat  = cv2.morphologyEx(binmask, cv2.MORPH_TOPHAT,   k)  # orig − opening
blackhat= cv2.morphologyEx(binmask, cv2.MORPH_BLACKHAT, k)  # closing − orig

show([binmask, erode, dilate, opening, closing, grad, tophat, blackhat],
     ['Бинарная маска', 'Erosion', 'Dilation', 'Opening', 'Closing',
      'Gradient (контуры)', 'TopHat', 'BlackHat'],
     figsize=(20, 5), gray_cmap='gray')

# ---------- 4.3 Как морфология улучшает контуры ----------
raw_cnts, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean_mask  = cv2.morphologyEx(binmask, cv2.MORPH_OPEN,  k, iterations=2)
clean_mask  = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, k, iterations=2)
clean_cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'Контуров без морфологии: {len(raw_cnts)}')
print(f'Контуров после морфологии: {len(clean_cnts)}')