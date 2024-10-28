import math
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


import os
from PIL import Image

lowest_common_multiple = 32


# input_folder = './base'
# output_folder = './process/base'
# crop_images(input_folder, output_folder)
# input_folder = './extended'
# output_folder = './process/extended'
# crop_images(input_folder, output_folder)


input_folder = './process_v2/extended'


def images_process(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png')):  # 支持的文件格式
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            class_num = 12
            temp_img = np.array(img)
            plt.imshow(temp_img)
            plt.axis('off')
            plt.show()
            new_data = torch.tensor(np.array(img), dtype=torch.int)  # Use long type for integer values
            new_data_shape = new_data.shape
            new_data = new_data.reshape(1, new_data_shape[0], new_data_shape[1])
            new_tensor = torch.zeros((class_num, new_data_shape[0], new_data_shape[1]), dtype=torch.float32)
            # 将原始 tensor 中的值映射到新 tensor
            for i in range(new_data_shape[0]):
                for j in range(new_data_shape[1]):
                    pixel_value = new_data[0, i, j].item()  # 获取像素值
                    new_tensor[pixel_value - 1, i, j] = 1.0  # 将对应索引位置设为 1
            pnz_path = img_path.replace('.png','.npz')
            loaded_npz = np.load(pnz_path)
            loaded_data_np = loaded_npz['new_tensor']
            loaded_data = torch.from_numpy(loaded_data_np)

            print(torch.equal(loaded_data,new_tensor))
images_process(input_folder)