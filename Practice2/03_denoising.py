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

# ---------- 3.1 Добавление шума ----------
def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_salt_pepper(img, amount=0.03):
    out = img.copy()
    n = int(amount * img.shape[0] * img.shape[1])
    ys = np.random.randint(0, img.shape[0], n)
    xs = np.random.randint(0, img.shape[1], n)
    out[ys[:n//2], xs[:n//2]] = 255          # соль
    out[ys[n//2:], xs[n//2:]] = 0            # перец
    return out

clean_img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
noisy_g   = add_gaussian_noise(clean_img, sigma=25)
noisy_sp  = add_salt_pepper(clean_img, amount=0.04)

# ---------- 3.2 Методы денойзинга ----------
def denoise_all(noisy):
    return {
        'Gaussian': cv2.GaussianBlur(noisy, (5, 5), 0),
        'Median'  : cv2.medianBlur(noisy, 5),
        'Bilateral': cv2.bilateralFilter(noisy, 9, 75, 75),
        'NLM'      : cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)
    }

res_g  = denoise_all(noisy_g)
res_sp = denoise_all(noisy_sp)

# ---------- 3.3 Оценка качества ----------
def score(ref, test):
    p = psnr(ref, test)
    s = ssim(ref, test, channel_axis=2, data_range=255)
    return p, s

print('=== Шум Гаусса (σ=25) ===')
print(f"{'Метод':<12}{'PSNR':>8}{'SSIM':>8}")
for name, im in res_g.items():
    p, s = score(clean_img, im)
    print(f'{name:<12}{p:8.2f}{s:8.4f}')

print('\n=== Соль и перец (4%) ===')
print(f"{'Метод':<12}{'PSNR':>8}{'SSIM':>8}")
for name, im in res_sp.items():
    p, s = score(clean_img, im)
    print(f'{name:<12}{p:8.2f}{s:8.4f}')

# Визуализация
show([clean_img, noisy_g, res_g['Gaussian'], res_g['Median'],
      res_g['Bilateral'], res_g['NLM']],
     ['Оригинал', 'Гауссов шум', 'Gaussian blur', 'Median blur',
      'Bilateral', 'NLM (fastNlMeans)'], figsize=(18, 4))

show([clean_img, noisy_sp, res_sp['Gaussian'], res_sp['Median'],
      res_sp['Bilateral'], res_sp['NLM']],
     ['Оригинал', 'Соль и перец', 'Gaussian blur', 'Median blur',
      'Bilateral', 'NLM'], figsize=(18, 4))