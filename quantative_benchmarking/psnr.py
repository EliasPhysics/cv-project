import os

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io


def calculate_psnr(image1, image2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image1 (numpy.ndarray): First image (ground truth).
        image2 (numpy.ndarray): Second image (predicted/reconstructed).

    Returns:
        float: PSNR score (higher values indicate better quality).
    """
    # Ensure the images are in the same shape
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((image1 - image2) ** 2)

    # Handle the case where MSE is zero (identical images)
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    max_pixel = 255.0 if image1.dtype == np.uint8 else 1.0
    psnr_score = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr_score



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
    psnr_score = calculate_psnr(image1, image2)

    print(f"PSNR: {psnr_score}")