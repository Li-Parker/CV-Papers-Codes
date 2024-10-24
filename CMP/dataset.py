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

transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((256,256)),
    transforms.Normalize(mean=0.5, std=0.5)
])
class CMP_dataset(data.Dataset):
    def __init__(self,imgs_path, annos_path):
        self.imgs_path = imgs_path
        self.annos_path = annos_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        pil_img = Image.open(img_path)
        pil_img = transforms(pil_img)

        anno_path = self.annos_path[index]
        anno_img = Image.open(anno_path)
        anno_img = anno_img.convert("RGB")
        pil_anno = transforms(anno_img)

        return pil_anno, pil_img

    def __len__(self):
        return len(self.imgs_path)


def get_dataset():
    train_imgs_path = glob.glob('./data/process/base/*.jpg')
    train_annos_path = glob.glob('./data/process/base/*.png')
    test_imgs_path = glob.glob('./data/process/extended/*.jpg')
    test_annos_path = glob.glob('./data/process/extended/*.png')

    train_dataset = CMP_dataset(train_imgs_path, train_annos_path)
    test_dataset = CMP_dataset(test_imgs_path, test_annos_path)
    BATCHSIZE = 16
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=BATCHSIZE,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=BATCHSIZE,
                                      shuffle=True)
    return train_dataloader, test_dataloader