import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm  # 导入 tqdm
from CMP.model.fcn import FCNs
from dataset import CMP_dataset, get_dataset

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
epochs = 100
for j in range(epochs):
    mymodel.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在评估时不计算梯度
        # 遍历测试集
        for i, test_item in enumerate(test_dataset):
            image, target = test_item
            image = image.to(device)
            target = target.to(device)

            output = mymodel(image)
            loss = criterion(output, target)

            # 显示图像
            case_show = output[0][0].cpu().detach().numpy()
            plt.imshow(case_show)
            plt.axis('off')
            plt.show()

            if i >= 0:  # 只显示第一条数据
                break  # 退出循环，只处理一条测试数据

    mymodel.train(True)
    # 使用 tqdm 包装 train_dataset 以显示进度条
    loss_total = 0
    loss_avg = 0
    with tqdm(total=len(train_dataset), desc=f'Epoch {j + 1}', unit='batch') as pbar:
        for i, item in enumerate(train_dataset):
            image, target = item
            image_temp = np.array(image[0])
            target_temp = np.array(target[0])
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = mymodel(image)

            loss = criterion(output, target)
            loss_total += loss
            loss_avg = loss_total/(i+1)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())  # 在进度条旁边显示当前损失
        print(loss_avg)

print("end---")
