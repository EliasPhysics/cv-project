import colorconvert as colconv
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import glob


def load_convert_lab(src_path,lab_path,print_freq=100,factor=0,print_size=False,target_shape_crop=None,norm_lab=False):
    """
    Converts RGB images to Lab and grayscale, then saves them as .npy files.

    Args:
        src_path (str): Directory containing source images.
            ###gray_path (str): Directory to save grayscale images.
        lab_path (str): Directory to save Lab images.
        print_freq (int, optional): Frequency of progress prints. Default is 100.
        factor (int, optional): Factor for resolution scaling. Default is 0.
        print_size (bool, optional): Whether to print the shape of Lab images. Default is False.
        target_shape (tuple, optional): Target shape for cropping. Default is None.
        norm_lab (bool, optional): Normalise the lab image to range [0,1]. Default is False.

    Returns:
        None
    """
    c = 0
    print("converting and saving")
    for i,file_path in enumerate(glob.glob(f"{src_path}/*.jpg")):  # Iterate over all JPG files
        if i%print_freq==0:
            print(f"progress : {i}")
        img = io.imread(file_path)  # Load image
        img_shape = img.shape

        if len(img_shape) != 3:
            continue

        if target_shape_crop is not None and (img_shape[0] < target_shape_crop[0] or img_shape[1] < target_shape_crop[1]):
            continue

        lab = colconv.rgb2lab_rgb2grey(img,factor=factor,target_shape=target_shape_crop,norm=norm_lab)

        if print_size:
            print(lab.shape)
        
        c += 1
        filename_img = str(c)
        np.save(lab_path+"/lab"+filename_img,lab)
        #np.save(gray_path+"/gray"+filename_img,gray)
    print("done")
    print(f"Number of converted files : {c}")

def find_smallest_dim(path,print_freq=100):
    # (H,W,C)
    min_H = 1000000
    min_W = 1000000

    for i,file_path in enumerate(glob.glob(f"{path}/*.jpg")):
        if i%print_freq == 0:
            print(f"progress : {i}")
        img = io.imread(file_path)
        img_shape = img.shape
        if len(img_shape) == 3:
            H,W,C = img_shape # img.shape
        else:
            H,W = img_shape # img.shape

        if H < min_H:
            min_H = H
        if W < min_W:
            min_W = W
    
    return min_H, min_W

import os
from collections import Counter

def find_most_common_shape(path, min_shape=(200, 200), print_freq=100):
    """
    Find the most common image shape that's larger than specified minimum dimensions
    
    Args:
        path: Directory containing images
        min_shape: Tuple (min_height, min_width) to filter shapes
        print_freq: Frequency of progress updates
        
    Returns:
        Tuple (height, width) of most common valid shape, or None if none found
    """
    shape_counter = Counter()
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    min_h, min_w = min_shape
    processed_count = 0
    
    # Collect valid image files
    file_paths = [
        os.path.join(path, f) for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not file_paths:
        print("No valid image files found.")
        return None
    
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths):
        try:
            img = io.imread(file_path)
            H, W = img.shape[:2]  # Get height/width regardless of channels
            
            # Only count shapes larger than minimum in both dimensions
            if H >= min_h and W >= min_w:
                shape_counter[(H, W)] += 1
                processed_count += 1
                
            # Progress reporting
            if i % print_freq == 0:
                print(f"Checked {i+1}/{total_files} files | "
                      f"Valid shapes found: {processed_count}")
                
        except Exception as e:
            print(f"Skipped {os.path.basename(file_path)} - Error: {str(e)}")
    
    if not shape_counter:
        print(f"No images larger than {min_shape} found.")
        return None
    
    # Get the most frequent shape from valid entries
    most_common = shape_counter.most_common(1)[0]
    print(f"Most common valid shape: {most_common[0]} (appeared {most_common[1]} times)")
    return most_common[0]

def check_size(img_path):
    """Returns size of image saved at img_path"""
    img = io.imread(img_path)

    return img.shape

def jpg_process(src_path,trg_path,print_freq=100,factor=0,target_shape_crop=None,norm_lab=False,to_size=None):
    c = 0
    print("converting and saving")
    for i,file_path in enumerate(os.listdir(src_path)):#glob.glob(f"{src_path}/*.jpg")):  # Iterate over all JPG files
        #os.listdir(src_path)
        
        if i%print_freq==0:
            print(f"progress : {i}")
        img = io.imread(src_path+"/"+file_path)  # Load image
        img_shape = img.shape

        if len(img_shape) != 3:
            continue


        if (target_shape_crop is not None) and (img_shape[0] < target_shape_crop[0] or img_shape[1] < target_shape_crop[1]):
            continue

        if (to_size is not None) and (img_shape[0] < to_size[0] or img_shape[1] < to_size[1]):
            continue

        if to_size is not None:
            img = colconv.resize_image(img,target_size=to_size)

        if factor > 0:
            img = colconv.transform.rescale(img, factor, anti_aliasing=True,multichannel=True)

        if target_shape_crop is not None and to_size is None:
            img = colconv.crop_to_size(img,target_shape_crop)

        c += 1
        io.imsave(trg_path+"/"+str(c)+"c.jpg",img)
      
        #np.save(gray_path+"/gray"+filename_img,gray)
    print("done")
    print(f"Number of converted files : {c}")

# test
#target_shape_crop = (309,232,3) # None
#load_convert_lab("datatreatment/rgb_test","datatreatment/gt_lab_test",factor=0.1,print_size=False,target_shape_crop=target_shape_crop)

# ## test
# loaded_lab = np.load("datatreatment/gt_lab_test/lab4.npy")
# colconv.display_lab(loaded_lab)

# Hmin, Wmin = find_smallest_dim("Caltech_WebFaces")

# print(f"smallest height : {Hmin}, smallest width : {Wmin}")

# not so usefull
#most_common_shape = find_most_common_shape("Caltech_WebFaces", print_freq=100)

#target_shape_crop = (256, 256, 3)
#load_convert_lab("Caltech_WebFaces","Lab_Caltech_WebFaces",print_size=False,target_shape_crop=target_shape_crop)
#jpg_process("Caltech_WebFaces","Lab_Caltech_WebFaces",target_shape_crop=target_shape_crop)
#to_size = (256,256)
#jpg_process("Caltech_WebFaces","Lab3",to_size=to_size)

#for i in range(2148):
    #imsize = check_size(f"Lab3/{i+1}.jpg")
    #if imsize != (256,256,3):
        #print(imsize)

to_size = (128,128)
src_path = "../../images/train" # training from 1m_faces_02, validation 1m_faces_03
jpg_process(src_path,"../data/mixed128",to_size=to_size)    