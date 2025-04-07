import numpy as np
import matplotlib.pyplot as plt
from skimage import color,transform


# lab_range = −128 to 127 

# Normalize LAB channels
L_min, L_max = 0, 100  # L channel range
a_min, a_max = -128, 127  # a channel range
b_min, b_max = -128, 127  # b channel range



def crop_to_size(img, shape_tpl):
    """Crops img to the shape of the given shape tuple, removing equal parts from both sides."""
    H, W, C = img.shape  # Original dimensions
    N, M, _ = shape_tpl  # Target dimensions

    # Ensure cropping is possible
    assert N <= H and M <= W, f"Target size ({N}, {M}) is larger than image size ({H}, {W})"

    # Compute how much to remove
    remove_H = H - N
    remove_W = W - M

    # Compute cropping indices
    top = remove_H // 2
    bottom = top + (remove_H % 2)  # Extra row if odd

    left = remove_W // 2
    right = left + (remove_W % 2)  # Extra column if odd

    # Crop the image
    img = img[top:H-bottom, left:W-right, :]

    return img


def resize_image(img, target_size=(256, 256)):
    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a numpy array")
    
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Determine the scaling factor to resize the shorter side to target size
    if h < w:
        new_h = target_h
        new_w = int(np.round(w * (target_h / h)))
    else:
        new_w = target_w
        new_h = int(np.round(h * (target_w / w)))
    
    # Resize with anti-aliasing
    resized = transform.resize(
        img,
        (new_h, new_w),
        anti_aliasing=True,
        preserve_range=True
    )
    
    # Calculate center crop indices (ensures within bounds)
    start_h = (new_h - target_h) // 2
    start_w = (new_w - target_w) // 2
    cropped = resized[
        start_h : start_h + target_h,
        start_w : start_w + target_w
    ]
    
    # Ensure output is exactly (256, 256) even if resized dimensions were slightly off due to rounding
    # This step is technically redundant due to the logic above but adds a safeguard
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        cropped = transform.resize(
            cropped,
            target_size,
            anti_aliasing=True,
            preserve_range=True
        )
    
    # Handle integer dtype
    if np.issubdtype(img.dtype, np.integer):
        cropped = np.round(cropped).astype(img.dtype)
    else:
        cropped = cropped.astype(img.dtype)
    
    return cropped

def norm_lab_img(lab_img):
    """Convert LAB image to normalized range [-1,1]."""
    lab_img[:, :, 0] = (lab_img[:, :, 0] - 0) / (100 - 0) * 2 - 1  # Scale L from [0,100] → [-1,1]
    lab_img[:, :, 1] = (lab_img[:, :, 1] + 128) / (127 + 128) * 2 - 1  # Scale a from [-128,127] → [-1,1]
    lab_img[:, :, 2] = (lab_img[:, :, 2] + 128) / (127 + 128) * 2 - 1  # Scale b from [-128,127] → [-1,1]

    return lab_img

def unnorm_lab_img(normalized_lab_img):
    """Convert LAB image from [-1,1] range back to original LAB range."""
    normalized_lab_img[:, :, 0] = ((normalized_lab_img[:, :, 0] + 1) / 2) * 100  # Convert L from [-1,1] → [0,100]
    normalized_lab_img[:, :, 1] = ((normalized_lab_img[:, :, 1] + 1) / 2) * (127 + 128) - 128  # Convert a from [-1,1] → [-128,127]
    normalized_lab_img[:, :, 2] = ((normalized_lab_img[:, :, 2] + 1) / 2) * (127 + 128) - 128  # Convert b from [-1,1] → [-128,127]
    
    return normalized_lab_img


def rgb2lab_rgb2grey(img,factor=0,norm=False,target_shape=None):
    """
    Convert RGB image to LAB and grayscale, with optional resolution reduction and normalization.
    Optional cropping to size given by target_shape=tuple(N,M,C)
    
    returns the LAB image (optionally normalized) and the greyscale image
    """
    if factor > 0:
        img = transform.rescale(img, factor, anti_aliasing=True,multichannel=True)
    if target_shape is not None:
        img = crop_to_size(img,target_shape)
    
    lab_img = color.rgb2lab(img)
    #grey_img = color.rgb2gray(img)

    if norm: 
        lab_img = norm_lab_img(lab_img)

    return lab_img#, grey_img

def display_lab(lab_img,save=False,filename="test_rgb_img.jpg"):
    """Display LAB image as RGB, with optional saving."""
    if np.min(lab_img) >= 0:
        lab_img = unnorm_lab_img(lab_img)

    rgb_img = color.lab2rgb(lab_img)

    if save:
        print("saving img")
        plt.imsave(filename, rgb_img)
        print("img saved")

    plt.imshow(rgb_img)
    plt.axis('off')  
    plt.show()
    
