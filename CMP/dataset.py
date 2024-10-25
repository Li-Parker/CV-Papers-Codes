import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

# # Define transforms for images and annotations separately
# image_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=0.5,std=0.5)
#     # Optionally add more transformations for images here (e.g., Resize, etc.)
# ])
#
# annotation_transforms = transforms.Compose([
#     transforms.ToTensor()  # Only convert to tensor without normalization
# ])

# Define transforms for images and annotations separately
image_transforms = transforms.Compose([
    transforms.ToTensor(),  # Apply normalization for images
])

# For annotations, just convert to tensor without normalization
def annotation_transforms(anno_img):
    new_data = torch.tensor(np.array(anno_img), dtype=torch.float)  # Use long type for integer values
    new_data_shape = new_data.shape
    new_data = new_data.reshape(1,new_data_shape[0],new_data_shape[1])
    return new_data


class CMP_dataset(data.Dataset):
    def __init__(self, imgs_path, annos_path):
        self.imgs_path = imgs_path
        self.annos_path = annos_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        pil_img = Image.open(img_path)
        pil_img = image_transforms(pil_img)  # Apply image transforms

        anno_path = self.annos_path[index]
        anno_img = Image.open(anno_path)
        pil_anno = annotation_transforms(anno_img)  # Apply annotation transforms

        return pil_img, pil_anno

    def __len__(self):
        return len(self.imgs_path)



def get_dataset():
    train_imgs_path = glob.glob('./data/process_v2/base/*.jpg')
    train_annos_path = glob.glob('./data/process_v2/base/*.png')
    test_imgs_path = glob.glob('./data/process_v2/extended/*.jpg')
    test_annos_path = glob.glob('./data/process_v2/extended/*.png')

    train_dataset = CMP_dataset(train_imgs_path, train_annos_path)
    test_dataset = CMP_dataset(test_imgs_path, test_annos_path)
    BATCHSIZE = 16
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=BATCHSIZE,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=True)
    return train_dataloader, test_dataloader