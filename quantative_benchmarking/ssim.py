import os

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io

def calculate_ssim(image1, image2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        image1 (numpy.ndarray): First image (ground truth).
        image2 (numpy.ndarray): Second image (predicted/reconstructed).

    Returns:
        float: SSIM score (higher values indicate greater similarity).
    """
    # Ensure the images are in the same shape
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Convert images to grayscale if they are in color
    if len(image1.shape) == 3:  # Color image (H, W, C)
        image1 = np.mean(image1, axis=2)
        image2 = np.mean(image2, axis=2)

    # Calculate SSIM
    ssim_score = ssim(image1, image2, data_range=image1.max() - image1.min())
    return ssim_score




# Example usage
if __name__ == "__main__":
    # Load two images (ensure they are in the same shape)
    os.chdir("..")
    image1 = io.imread("1.png")
    image2 = io.imread("2.png")

    # Convert images to float32 if they are in uint8 format
    if image1.dtype == np.uint8:
        image1 = image1.astype(np.float32) / 255.0
    if image2.dtype == np.uint8:
        image2 = image2.astype(np.float32) / 255.0

    # Calculate SSIM and PSNR
    ssim_score = calculate_ssim(image1, image2)

    print(f"SSIM: {ssim_score:.4f}")