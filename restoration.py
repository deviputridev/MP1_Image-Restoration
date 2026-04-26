import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# UTILITIES
# =========================
def clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def pad_image(img, pad):
    return np.pad(img, pad, mode='edge')

# =========================
# MEDIAN FILTER
# =========================
def median_filter(img, ksize=3):
    pad = ksize // 2
    padded = pad_image(img, pad)
    h, w = img.shape
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+ksize, j:j+ksize]
            output[i, j] = np.median(window)

    return output

# =========================
# GAUSSIAN FILTER
# =========================
def gaussian_kernel(ksize=5, sigma=1.0):
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter(img, ksize=5, sigma=1.0):
    kernel = gaussian_kernel(ksize, sigma)
    pad = ksize // 2
    padded = pad_image(img, pad)
    h, w = img.shape
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+ksize, j:j+ksize]
            output[i, j] = np.sum(window * kernel)

    return clip(output)

# =========================
# HISTOGRAM EQUALIZATION
# =========================
def histogram_equalization(img):
    hist = np.zeros(256)
    for pixel in img.flatten():
        hist[pixel] += 1

    cdf = np.cumsum(hist)
    cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_norm = cdf_norm.astype(np.uint8)

    return cdf_norm[img]

# =========================
# SHARPENING
# =========================
def unsharp_mask(img, ksize=5, sigma=1.0, alpha=1.5):
    blurred = gaussian_filter(img, ksize, sigma)
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img.astype(np.float32) + alpha * mask
    return clip(sharpened)

# =========================
# MAIN
# =========================
def main():
    img = cv2.imread("input/lena_noisy.png")
    if img is None:
        print("Image not found!")
        return

    # Split channels
    b, g, r = cv2.split(img)

    # ---------- 1. MEDIAN ----------
    b_med = median_filter(b)
    g_med = median_filter(g)
    r_med = median_filter(r)
    median_img = cv2.merge([b_med, g_med, r_med])
    cv2.imwrite("output/1_median.png", median_img)

    # ---------- 2. GAUSSIAN ----------
    b_gau = gaussian_filter(b_med)
    g_gau = gaussian_filter(g_med)
    r_gau = gaussian_filter(r_med)
    gaussian_img = cv2.merge([b_gau, g_gau, r_gau])
    cv2.imwrite("output/2_gaussian.png", gaussian_img)

    # ---------- 3. HISTOGRAM ----------
    b_he = histogram_equalization(b_gau)
    g_he = histogram_equalization(g_gau)
    r_he = histogram_equalization(r_gau)
    he_img = cv2.merge([b_he, g_he, r_he])
    cv2.imwrite("output/3_histogram_equalization.png", he_img)

    # ---------- 4. SHARPEN ----------
    b_sharp = unsharp_mask(b_he)
    g_sharp = unsharp_mask(g_he)
    r_sharp = unsharp_mask(r_he)
    final_img = cv2.merge([b_sharp, g_sharp, r_sharp])
    cv2.imwrite("output/4_sharpening.png", final_img)

    # ---------- VISUALIZATION ----------
    titles = ["Input", "Median", "Gaussian", "Hist Eq", "Sharpen"]
    images = [img, median_img, gaussian_img, he_img, final_img]

    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(titles[i])
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()