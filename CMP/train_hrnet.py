import math

import pandas
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm  # 导入 tqdm
from CMP.model.fcn import FCNs
from CMP.model.hrnet.hrnet import HighResolutionNet
from dataset import CMP_dataset, get_dataset
# --- HRNET_32 --- #
hrnet32 = {
    'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK',
               'FUSE_METHOD': 'SUM'},
    'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [32, 64], 'BLOCK': 'BASIC',
               'FUSE_METHOD': 'SUM'},
    'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [32, 64, 128],
               'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
    'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [32, 64, 128, 256],
               'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'}}





CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')
num_classes = 12
criterion = nn.BCEWithLogitsLoss()

mymodel = HighResolutionNet(cfg=hrnet32, options=None, num_classes=num_classes).to(device)
mymodel.load_state_dict(torch.load('./saved_hrnet/best_model_epoch33_loss0.0211.pth'))

optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)

train_dataset, test_dataset = get_dataset()
epoch_loss = 0.0
batches = 0
verbose = 10
logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
epochs = 21
best_loss = 100

loss_record = []
for j in range(33, epochs+33):
    mymodel.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在评估时不计算梯度
        # 遍历测试集
        for i, test_item in enumerate(test_dataset):
            image, target = test_item
            image = image.to(device)
            target = target.to(device)

            output = mymodel.forward(image)
            loss = criterion(output, target)
            y_shape = output.shape
            y = torch.argmax(output, dim=1) + torch.tensor(1.0)
            y = y.reshape(y_shape[0], 1, y_shape[2], y_shape[3])
            # 显示图像
            case_show = y[0][0].cpu().detach().numpy()
            plt.imshow(case_show)
            plt.axis('off')
            plt.show()

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
            output = mymodel.forward(image)

            loss = criterion(output, target)
            loss_total += loss.item()
            loss_avg = loss_total/(i+1)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())  # 在进度条旁边显示当前损失
            if i%100 == 0:
                print(loss_avg)
                loss_record.append((j,i,loss_avg))
                # 更新最佳损失并保存模型
                if math.fabs(loss_avg) < best_loss:
                    best_loss = math.fabs(loss_avg)
                    model_path = f'./saved_hrnet/best_model_epoch{j + 1}_num{i+1}_loss{best_loss:.4f}.pth'  # 使用epoch和loss更新文件名
                    torch.save(mymodel.state_dict(), model_path)
                    print(f'Model saved to {model_path} with loss: {best_loss:.4f}')
            else:
                continue
    # 更新最佳损失并保存模型
    model_path = f'./saved_hrnet/best_model_epoch{j + 1}_loss{best_loss:.4f}.pth'  # 使用epoch和loss更新文件名
    torch.save(mymodel.state_dict(), model_path)
    loss_record_dp = pandas.DataFrame(loss_record)
    loss_record_dp.to_csv("./saved_hrnet/loss_record.csv", mode="a")
    loss_record = []
print("end---")


