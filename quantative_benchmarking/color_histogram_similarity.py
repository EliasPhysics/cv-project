import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


def calculate_color_histogram(image, bins=8):
    """
    Calculate the color histogram of an image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        bins (int): Number of bins for each channel.

    Returns:
        numpy.ndarray: Flattened color histogram.
    """
    # Convert the image to HSV color space (optional, can use RGB as well)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for each channel
    hist_h = cv2.calcHist([hsv_image], [0], None, [bins], [0, 180])  # Hue
    hist_s = cv2.calcHist([hsv_image], [1], None, [bins], [0, 256])  # Saturation
    hist_v = cv2.calcHist([hsv_image], [2], None, [bins], [0, 256])  # Value

    # Normalize the histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # Concatenate the histograms into a single feature vector
    hist = np.hstack([hist_h, hist_s, hist_v])
    return hist


def color_histogram_similarity(image1, image2, bins=8):
    """
    Calculate the color histogram similarity between two images.

    Args:
        image1 (numpy.ndarray): First image in BGR format.
        image2 (numpy.ndarray): Second image in BGR format.
        bins (int): Number of bins for each channel.

    Returns:
        float: Similarity score (higher values indicate greater similarity).
    """
    # Calculate histograms for both images
    hist1 = calculate_color_histogram(image1, bins)
    hist2 = calculate_color_histogram(image2, bins)

    # Compute cosine similarity between the histograms
    similarity = cosine_similarity(hist1.reshape(1, -1), hist2.reshape(1, -1))
    return similarity[0][0]


# Example usage
if __name__ == "__main__":

    # Load two images
    os.chdir("..")
    image1 = cv2.imread("1.png")
    image2 = cv2.imread("2.png")
    # Resize images to the same dimensions (optional but recommended)
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))

    # Calculate color histogram similarity
    similarity_score = color_histogram_similarity(image1, image2)
    print(f"Color Histogram Similarity: {similarity_score:.4f}")