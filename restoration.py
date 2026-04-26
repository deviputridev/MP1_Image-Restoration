import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

def clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def pad_image(img, pad):
    return np.pad(img, pad, mode='edge')

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def compute_histogram(channel):
    hist = np.zeros(256, dtype=np.int64)
    for pixel in channel.flatten():
        hist[pixel] += 1
    return hist

# MEDIAN FILTER
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

# GAUSSIAN FILTER
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

# HISTOGRAM EQUALIZATION
def histogram_equalization(img):
    hist = np.zeros(256)
    for pixel in img.flatten():
        hist[pixel] += 1
    cdf = np.cumsum(hist)
    cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_norm = cdf_norm.astype(np.uint8)
    return cdf_norm[img]

# SHARPENING: Unsharp Masking
def unsharp_mask(img, ksize=5, sigma=1.0, alpha=1.5):
    blurred = gaussian_filter(img, ksize, sigma)
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img.astype(np.float32) + alpha * mask
    return clip(sharpened)

# VISUALIZATION 1: Pipeline Image Comparison
def plot_pipeline_images(images_dict, save_path="output/pipeline_comparison.png"):
    titles = list(images_dict.keys())
    imgs   = list(images_dict.values())
    n = len(imgs)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    fig.suptitle("Pipeline Restorasi: Perbandingan Visual Tiap Tahap",
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, title, img in zip(axes, titles, imgs):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Pipeline image comparison  -> {save_path}")

# VISUALIZATION 2: RGB Histogram per Pipeline Stage
def plot_histogram_comparison(images_dict, save_path="output/histogram_comparison.png"):
    stages  = list(images_dict.keys())
    n       = len(stages)
    ch_info = [("Blue", 0, "#3a7ebf"), ("Green", 1, "#3aab5c"), ("Red", 2, "#d94f3d")]

    fig, axes = plt.subplots(n, 3, figsize=(14, 3.2 * n))
    fig.suptitle("Histogram RGB per Tahap Pipeline Restorasi",
                 fontsize=14, fontweight='bold', y=1.005)

    for row, (stage, img_bgr) in enumerate(images_dict.items()):
        for col, (ch_name, ch_idx, color) in enumerate(ch_info):
            ax    = axes[row, col]
            data  = img_bgr[:, :, ch_idx]
            hist  = compute_histogram(data)
            x     = np.arange(256)

            ax.fill_between(x, hist, alpha=0.50, color=color)
            ax.plot(x, hist, color=color, linewidth=0.9)

            mean_v = float(np.mean(data))
            std_v  = float(np.std(data))
            ax.axvline(mean_v, color='black', linestyle='--',
                       linewidth=1.0, label=f'μ={mean_v:.1f}')

            ax.set_xlim(0, 255)
            ax.set_ylim(0, hist.max() * 1.15 if hist.max() > 0 else 1)
            ax.tick_params(labelsize=7)
            ax.grid(axis='y', linestyle='--', alpha=0.35)
            ax.legend(fontsize=6.5, loc='upper right')
            ax.text(0.03, 0.88, f'σ={std_v:.1f}', transform=ax.transAxes,
                    fontsize=6.5, color='black')

            if col == 0:
                ax.set_ylabel(stage, fontsize=9, fontweight='bold')
            if row == 0:
                ax.set_title(f"Channel {ch_name}", fontsize=10, fontweight='bold')
            if row == n - 1:
                ax.set_xlabel("Intensity (0–255)", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Histogram comparison       -> {save_path}")

# VISUALIZATION 3: CDF Before vs After Histogram EQ
def plot_cdf_comparison(before_bgr, after_bgr,
                        save_path="output/cdf_comparison.png"):
    ch_info = [("Blue", 0, "#3a7ebf"),
               ("Green", 1, "#3aab5c"),
               ("Red",   2, "#d94f3d")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("CDF: Sebelum vs Sesudah Histogram Equalization",
                 fontsize=12, fontweight='bold')

    for col, (ch_name, ch_idx, color) in enumerate(ch_info):
        ax = axes[col]
        for label, img_bgr, ls in [("Before HEQ", before_bgr, '--'),
                                    ("After HEQ",  after_bgr,  '-')]:
            hist     = compute_histogram(img_bgr[:, :, ch_idx])
            cdf      = np.cumsum(hist)
            cdf_pct  = cdf / cdf[-1] * 100
            ax.plot(cdf_pct, color=color, linestyle=ls, linewidth=1.6, label=label)

        ax.set_title(f"Channel {ch_name}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Intensity (0–255)", fontsize=8)
        ax.set_ylabel("CDF (%)", fontsize=8)
        ax.set_xlim(0, 255);  ax.set_ylim(0, 105)
        ax.legend(fontsize=8);  ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"CDF comparison             -> {save_path}")

# MAIN
def main():
    img = cv2.imread("input/lena_noisy.png")
    if img is None:
        print("Image not found!")
        return

    original = cv2.imread("input/lena_ori.png")

    b, g, r = cv2.split(img)

    # Step 1: Median Filter
    b_med = median_filter(b)
    g_med = median_filter(g)
    r_med = median_filter(r)
    median_img = cv2.merge([b_med, g_med, r_med])
    cv2.imwrite("output/1_median.png", median_img)

    # Step 2: Gaussian Filter
    b_gau = gaussian_filter(b_med)
    g_gau = gaussian_filter(g_med)
    r_gau = gaussian_filter(r_med)
    gaussian_img = cv2.merge([b_gau, g_gau, r_gau])
    cv2.imwrite("output/2_gaussian.png", gaussian_img)

    # Step 3: Histogram Equalization
    b_he = histogram_equalization(b_gau)
    g_he = histogram_equalization(g_gau)
    r_he = histogram_equalization(r_gau)
    he_img = cv2.merge([b_he, g_he, r_he])
    cv2.imwrite("output/3_histogram_equalization.png", he_img)

    # Step 4: Unsharp Masking (Sharpening)
    b_sharp = unsharp_mask(b_he)
    g_sharp = unsharp_mask(g_he)
    r_sharp = unsharp_mask(r_he)
    final_img = cv2.merge([b_sharp, g_sharp, r_sharp])
    cv2.imwrite("output/4_sharpening.png", final_img)
    cv2.imwrite("output/lena_restored.png", final_img)

    # PSNR Report
    print("\n" + "="*52)
    print("  PSNR Report (vs original)")
    print("="*52)
    if original is not None:
        stages_psnr = [
            ("Input (Noisy)",       img),
            ("Step 1 - Median",     median_img),
            ("Step 2 - Gaussian",   gaussian_img),
            ("Step 3 - Hist. EQ",   he_img),
            ("Step 4 - Sharpening", final_img),
        ]
        for name, s_img in stages_psnr:
            print(f"  {name:<28}: {psnr(s_img, original):.2f} dB")
    print("="*52 + "\n")

    # Stages dict for plots
    pipeline_stages = {
        "Input\n(Noisy)":    img,
        "Step 1\nMedian":    median_img,
        "Step 2\nGaussian":  gaussian_img,
        "Step 3\nHist. EQ":  he_img,
        "Step 4\nSharpen":   final_img,
    }
    if original is not None:
        pipeline_stages["Original"] = original

    # All Figure
    plot_pipeline_images(pipeline_stages)
    plot_histogram_comparison(pipeline_stages)
    plot_cdf_comparison(gaussian_img, he_img)

    print("And yap, done!")

if __name__ == "__main__":
    main()
