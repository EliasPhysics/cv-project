import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PairedNumpyDataset(Dataset):
    def __init__(self, input_paths, gt_paths):
        self.input_paths = input_paths  # List of input .npy file paths
        self.gt_paths = gt_paths  # List of ground truth .npy file paths

    def __getitem__(self, index):
        # Load input image from .npy file
        input_img = np.load(self.input_paths[index])  # Shape: (N, M, 1)
        input_img = np.expand_dims(input_img, axis=-1)
        input_img = np.transpose(input_img, (2, 0, 1))  # Convert to (1, N, M)
        input_img = torch.from_numpy(input_img).float()

        # Load ground truth image from .npy file
        gt_img = np.load(self.gt_paths[index])  # Shape: (N, M, 3)
        gt_img = np.transpose(gt_img, (2, 0, 1))  # Convert to (3, N, M)
        gt_img = torch.from_numpy(gt_img).float()

        return input_img, gt_img

    def __len__(self):
        return len(self.input_paths)

def files_names_list(N,input_path,gt_path,suffix=".npy"):
    input_name_list = []
    gt_name_list = []
    for i in range(N):
        im_num = i+1
        input_name_list.append(input_path+"gray"+str(im_num)+suffix)
        gt_name_list.append(gt_path+"lab"+str(im_num)+suffix)

    return input_name_list, gt_name_list



# Example Usage
input_paths, gt_paths = files_names_list(5,"datatreatment/input_gray_test/","datatreatment/gt_lab_test/")


dataset = PairedNumpyDataset(input_paths, gt_paths)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Load in batches

# Iterate through the DataLoader
for batch in dataloader:
    input_batch, gt_batch = batch
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

def torch_gray_to_numpy(tensor):
    """
    Converts a PyTorch grayscale image tensor (1, H, W) to a NumPy array (H, W).
    
    Args:
        tensor (torch.Tensor): A PyTorch tensor of shape (1, H, W)
    
    Returns:
        np.ndarray: A NumPy array of shape (H, W)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")

    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError("Input tensor must have shape (1, H, W)")

    return tensor.squeeze(0).cpu().numpy()

# import colorconvert as colconv

# lab_img = torch_to_numpy(gt_batch[0])

# colconv.display_lab(lab_img)



# gray_img = torch_gray_to_numpy(input_batch[0])

# import matplotlib.pyplot as plt

# gray = plt.imshow(gray_img,cmap="gray")
# plt.show()