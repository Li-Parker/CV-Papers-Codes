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
import logging

from CMP.model.fcn import FCNs
from dataset import CMP_dataset,get_dataset
CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')
num_classes = 3
criterion = nn.BCEWithLogitsLoss()

mymodel = FCNs(num_classes).to(device)
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)

train_dataset, test_dataset = get_dataset()
epoch_loss = 0.0
batches = 0
verbose = 10
logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)

for i, item in enumerate(train_dataset):
    image, target = item
    if CUDA:
        image = image.to(device)
        target = target.to(device)
    optimizer.zero_grad()
    output = mymodel(image)

    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    batches += 1
    print(i)
print("end---")
