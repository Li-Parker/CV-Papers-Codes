import argparse
import torch.backends.cudnn as cudnn
from CMP_V2.data_v2.get_data_set import get_data
import torch
import data_v2.utils2.misc as misc
from configs import get_args_parser
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from model.fcn import FCNs
import logging
from tqdm import tqdm  # 导入 tqdm
import math
import pandas


def main():
    # distribution
    misc.init_distributed_mode(args)
    cudnn.benchmark = True
    CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if CUDA else 'cpu')
    num_classes = 10
    criterion = nn.CrossEntropyLoss()
    mymodel = FCNs(num_classes).to(device)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)

    train_set, val_set, ignore_index = get_data(args)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=sampler_val, num_workers=args.num_workers, shuffle=False)

    epoch_loss = 0.0
    batches = 0
    verbose = 10
    logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    epochs = 101
    # 训练结束后保存模型
    model_path = 'model.pth'  # 指定保存的路径和文件名
    torch.save(mymodel.state_dict(), model_path)
    best_loss = 100

    loss_record = []
    for j in range(epochs):

        mymodel.train(True)
        # 使用 tqdm 包装 train_dataset 以显示进度条
        loss_total = 0
        loss_avg = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {j + 1}', unit='batch') as pbar:
            for i, data in enumerate(train_loader):
                # if data_iter_step % accum_iter == 0:
                #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

                inputs = data["images"].to(device, dtype=torch.float32)
                mask = data["masks"].to(device, dtype=torch.int64)
                optimizer.zero_grad()
                output = mymodel.forward(inputs)
                loss = criterion(output, mask)
                loss_total += loss.item()
                loss_avg = loss_total / (i + 1)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())  # 在进度条旁边显示当前损失
                if i % 20 == 0:
                    loss_record.append((j, i, loss_avg))
                else:
                    continue
        # 更新最佳损失并保存模型
        model_path = f'./saved/best_model_epoch{j + 1}_loss{loss_avg:.4f}.pth'  # 使用epoch和loss更新文件名
        if j % 2 == 0:
            torch.save(mymodel.state_dict(), model_path)
        loss_record_dp = pandas.DataFrame(loss_record)
        loss_record_dp.to_csv("loss_record.csv", mode="a")
        loss_record = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()