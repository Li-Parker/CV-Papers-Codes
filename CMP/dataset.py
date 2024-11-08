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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# For annotations, just convert to tensor without normalization
def annotation_transforms(anno_img):
    class_num = 12
    new_data = torch.tensor(np.array(anno_img), dtype=torch.int)  # Use long type for integer values
    new_data_shape = new_data.shape
    new_data = new_data.reshape(1,new_data_shape[0],new_data_shape[1])
    new_tensor = torch.zeros((class_num, new_data_shape[0], new_data_shape[1]), dtype=torch.float32)
    # 将原始 tensor 中的值映射到新 tensor
    for i in range(new_data_shape[0]):
        for j in range(new_data_shape[1]):
            pixel_value = new_data[0, i, j].item()  # 获取像素值
            new_tensor[pixel_value - 1, i, j] = 1.0  # 将对应索引位置设为 1
    return new_tensor
# def annotation_transforms(anno_img):
#     class_num = 12
#     new_data = torch.tensor(np.array(anno_img), dtype=torch.int64)  # 使用 int64 类型
#     new_data_shape = new_data.shape
#     new_data = new_data.unsqueeze(0)  # 在第一个维度添加维度，使形状为 (1, H, W)
#
#     # 创建新 tensor
#     new_tensor = torch.zeros((class_num, new_data_shape[0], new_data_shape[1]), dtype=torch.float32)
#
#     # 使用向量化操作，将像素值映射到新 tensor
#     # new_data[0] 是一个 (H, W) 的 tensor，使用 .long() 转换为长整型
#     new_tensor[new_data[0] - 1] = 1.0
#
#     return new_tensor

class CMP_dataset(data.Dataset):
    def __init__(self, imgs_path, annos_path):
        self.imgs_path = imgs_path
        self.annos_path = annos_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        pil_img = Image.open(img_path)
        pil_img = image_transforms(pil_img)  # Apply image transforms

        anno_path = self.annos_path[index]
        anno_img = np.load(anno_path)['new_tensor']
        pil_anno = torch.from_numpy(anno_img)  # Apply annotation transforms

        return pil_img, pil_anno

    def __len__(self):
        return len(self.imgs_path)



def get_dataset():
    train_imgs_path = glob.glob('./data/process_v2/base/*.jpg')
    train_annos_path = glob.glob('./data/process_v2/base/*.npz')
    test_imgs_path = glob.glob('./data/process_v2/extended/*.jpg')
    test_annos_path = glob.glob('./data/process_v2/extended/*.npz')

    train_dataset = CMP_dataset(train_imgs_path, train_annos_path)
    test_dataset = CMP_dataset(test_imgs_path, test_annos_path)
    BATCHSIZE = 4
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=BATCHSIZE,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=True)
    return train_dataloader, test_dataloader