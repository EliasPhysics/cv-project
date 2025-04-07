import colorconvert as colconv
from skimage import io
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

def check_size(img_path):
    """Returns size of image saved at img_path"""
    img = io.imread(img_path)

    return img.shape

def jpg_process(src_path,trg_path,print_freq=100,factor=0,target_shape_crop=None,norm_lab=False,to_size=None):
    c = 0
    print("converting and saving")
    for i,file_path in enumerate(glob.glob(f"{src_path}/*.jpg")):  # Iterate over all JPG files
        #os.listdir(src_path)
        if i%print_freq==0:
            print(f"progress : {i}")
        img = io.imread(file_path)  # Load image
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
        io.imsave(trg_path+"/"+str(c)+".jpg",img)
      
        #np.save(gray_path+"/gray"+filename_img,gray)
    print("done")
    print(f"Number of converted files : {c}")


to_size = (128,128)
src_path = "../1m_faces_03" # training from 1m_faces_02, validation 1m_faces_03
jpg_process(src_path,"data/val_faces128",to_size=to_size)    