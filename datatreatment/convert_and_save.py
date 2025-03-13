import colorconvert as colconv
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def load_convert(N,src_path,gray_path,lab_path,print_freq=100,factor=0,print_size=False,target_shape=None):
    print("converting and saving")
    for i in range(N):
        filename_img = str(i+1)
        img = io.imread(src_path+filename_img+".jpg")
        lab, gray = colconv.rgb2lab_rgb2grey(img,factor=factor,target_shape=target_shape)

        if print_size:
            print(lab.shape)
        
        np.save(lab_path+"lab"+filename_img,lab)
        np.save(gray_path+"gray"+filename_img,gray)

        if i%print_freq==0:
            print(f"progress : {i}")

    print("done")

# target_shape = (309,232,3) # None
# load_convert(5,"datatreatment/rgb_test/","datatreatment/input_gray_test/","datatreatment/gt_lab_test/",factor=0.1,print_size=False,target_shape=target_shape)

# ## test
# loaded_lab = np.load("datatreatment/gt_lab_test/lab4.npy")
# colconv.display_lab(loaded_lab)