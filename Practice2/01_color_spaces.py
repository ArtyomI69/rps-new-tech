import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------- Загрузка изображений ----------
img1 = cv2.imread('img1.png')            # 3 ЗЕЛЁНЫХ яблока
img2 = cv2.imread('img2.png')            # цветок крупным планом
if img1 is None or img2 is None:
    raise FileNotFoundError('Положите img1.png и img2.png рядом со скриптом')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

def show(images, titles, figsize=(16, 5), gray_cmap='gray'):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1: axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap=gray_cmap) if im.ndim == 2 else ax.imshow(im)
        ax.set_title(t, fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ---------- 1.1 RGB → HSV → RGB ----------
hsv      = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv_back = cv2.cvtColor(hsv,  cv2.COLOR_HSV2BGR)

# ---------- 1.2 RGB → LAB → RGB ----------
lab      = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
lab_back = cv2.cvtColor(lab,  cv2.COLOR_LAB2BGR)

# ---------- 1.3 RGB → YCrCb → RGB ----------
ycrcb      = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
ycrcb_back = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

show([img1_rgb,
      cv2.cvtColor(hsv,   cv2.COLOR_HSV2RGB),
      cv2.cvtColor(lab,   cv2.COLOR_LAB2RGB),
      cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB),
      cv2.cvtColor(hsv_back, cv2.COLOR_BGR2RGB)],
     ['Оригинал (RGB)',
      'HSV (H=тон, S=насыщ., V=яркость)',
      'LAB (L=светлота, a,b=цвет)',
      'YCrCb (Y=яркость, Cr/Cb=цветность)',
      'HSV → RGB (обратно)'])

diff_hsv = np.abs(img1.astype(int) - hsv_back.astype(int)).mean()
print('Средняя ошибка HSV → RGB:', diff_hsv)   # ~0 → преобразование без потерь

# ---------- 1.4 Сегментация ЗЕЛЁНЫХ яблок в HSV ----------
# OpenCV: H ∈ [0,179]; зелёный ≈ 35..85
lower_green = np.array([35,  60,  40])     # S,V подобраны с запасом
upper_green = np.array([85, 255, 255])

mask_raw = cv2.inRange(hsv, lower_green, upper_green)

# Чистим маску морфологией (убираем блики и мелкий шум)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN,  kernel, iterations=1)
mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, kernel, iterations=2)

# Наложим маску на оригинал для наглядности
segmented = cv2.bitwise_and(img1_rgb, img1_rgb, mask=mask)

# Покажем также сами каналы HSV — видно, какой канал «отвечает» за зелёный
h_ch, s_ch, v_ch = cv2.split(hsv)

show([img1_rgb, mask_raw, mask, segmented],
     ['Оригинал (зелёные яблоки)',
      'Маска зелёного (до морфологии)',
      'Маска зелёного (после морфологии)',
      'Результат сегментации'],
     figsize=(16, 4))

show([h_ch, s_ch, v_ch],
     ['Канал H (тон)', 'Канал S (насыщенность)', 'Канал V (яркость)'],
     figsize=(14, 4))

# ---------- 1.5 Совет по подбору порогов ----------
# Если яблоки не выделились полностью/захватили фон — подберите H вручную:
#   trackbar-скрипт ниже (раскомментируйте и запустите отдельно)
#
# cv2.namedWindow('tune')
# cv2.createTrackbar('H_min','tune',35,179,lambda x: None)
# cv2.createTrackbar('H_max','tune',85,179,lambda x: None)
# while True:
#     lo = np.array([cv2.getTrackbarPos('H_min','tune'), 60, 40])
#     hi = np.array([cv2.getTrackbarPos('H_max','tune'),255,255])
#     m  = cv2.inRange(hsv, lo, hi)
#     cv2.imshow('tune', m)
#     if cv2.waitKey(1) & 0xFF == 27: break
# cv2.destroyAllWindows()