import cv2
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000


def calculate_cie2000_difference(image1, image2):
    """
    Calculate the average CIEDE2000 color difference between two images.

    Args:
        image1 (numpy.ndarray): First image in BGR format.
        image2 (numpy.ndarray): Second image in BGR format.

    Returns:
        float: Average CIEDE2000 difference (lower values indicate greater similarity).
    """
    # Convert images to Lab color space
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2Lab)

    # Initialize a list to store CIEDE2000 differences
    differences = []

    # Iterate over each pixel and compute CIEDE2000
    for i in range(lab1.shape[0]):
        for j in range(lab1.shape[1]):
            # Extract Lab values for the current pixel
            l1, a1, b1 = lab1[i, j]
            l2, a2, b2 = lab2[i, j]

            # Create LabColor objects
            color1 = LabColor(l1, a1, b1)
            color2 = LabColor(l2, a2, b2)

            # Compute CIEDE2000 difference
            delta_e = delta_e_cie2000(color1, color2)
            differences.append(delta_e)

    # Compute the average CIEDE2000 difference
    average_difference = np.mean(differences)
    return average_difference


# Example usage
if __name__ == "__main__":
    # Load two images
    image1 = cv2.imread("image1.jpg")
    image2 = cv2.imread("image2.jpg")

    # Resize images to the same dimensions (optional but recommended)
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))

    # Calculate CIEDE2000 difference
    cie2000_diff = calculate_cie2000_difference(image1, image2)
    print(f"Average CIEDE2000 Difference: {cie2000_diff:.4f}")