import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io,color
import colorconvert as colconv

class PairedNumpyDataset(Dataset):
    def __init__(self, gt_paths,img_format="jpg"):
        #self.input_paths = input_paths  # List of input .npy file paths
        self.gt_paths = gt_paths  # List of ground truth .npy file paths
        self.img_format = img_format

    def __getitem__(self, index):
        # Load input image from .npy file
        # input_img = np.load(self.input_paths[index])  # Shape: (N, M, 1)
        # input_img = np.expand_dims(input_img, axis=-1)
        # input_img = np.transpose(input_img, (2, 0, 1))  # Convert to (1, N, M)
        # input_img = torch.from_numpy(input_img).float()

        # Load ground truth image from .npy file
        if self.img_format == "npy":
            gt_img = np.load(self.gt_paths[index])  # Shape: (N, M, 3)
            gt_img = np.transpose(gt_img, (2, 0, 1))  # Convert to (3, N, M)
            gt_img = torch.from_numpy(gt_img).float()
        else:
            gt_img = io.imread(self.gt_paths[index])
            gt_img = color.rgb2lab(gt_img)
            gt_img = colconv.norm_lab_img(gt_img)
            gt_img = np.transpose(gt_img, (2, 0, 1))
            gt_img = torch.from_numpy(gt_img).float()


        L = gt_img[0].unsqueeze(0)
        ab = gt_img[1:]

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.gt_paths)

def files_names_list(N,gt_path,prefix="lab",suffix=".npy"):
    # input_name_list = []
    gt_name_list = []
    for i in range(N):
        im_num = i+1
        # input_name_list.append(input_path+"/gray"+str(im_num)+suffix)
        gt_name_list.append(gt_path+"/"+prefix+str(im_num)+suffix)

    return gt_name_list



# Example Usage
gt_paths = files_names_list(5,"datatreatment/gt_lab_test")


dataset = PairedNumpyDataset(gt_paths,img_format="npy")
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)  # Load in batches

# Iterate through the DataLoader
for batch in dataloader:
    #input_batch, gt_batch = batch
    input_batch, gt_batch = batch['L'], batch['ab']
    print(f"Input batch shape: {input_batch.shape}")  # (batch_size, 1, N, M)
    print(f"GT batch shape: {gt_batch.shape}")  # (batch_size, 3, N, M)

### 
def torch_to_numpy(img_tensor):
    """
    Converts a PyTorch image tensor (C, H, W) to a NumPy array (H, W, C).

    Args:
        img_tensor (torch.Tensor): Input tensor with shape (C, H, W).

    Returns:
        np.ndarray: Converted NumPy array with shape (H, W, C).
    """
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")

    if img_tensor.ndim != 3:
        raise ValueError("Expected a 3D tensor (C, H, W), but got shape {}".format(img_tensor.shape))

    img_numpy = img_tensor.permute(1, 2, 0).cpu().numpy()  # Change (C, H, W) -> (H, W, C)
    return img_numpy

