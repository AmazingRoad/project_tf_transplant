import argparse
import time
import random
from torch.utils.data import DataLoader
from core.data.utils import *
from functools import partial
import os
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from core.models.ffn import FFN
from core.data import BatchCreator

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument(
    '-r', '--resume', metavar='PATH',
    help='Path to pretrained model state dict or a compiled and saved '
         'ScriptModule from which to resume training.'
)
parser.add_argument(
    '--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).'
)

parser.add_argument('-d', '--data', type=str, default='./data.h5', help='training data')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--delta', default=(8, 8, 8), help='delta offset')
parser.add_argument('--input_size', default=(33, 33, 33), help='input size')
parser.add_argument('--clip_grad_thr', type=float, default=0.5, help='grad clip threshold')
parser.add_argument('--save_path', type=str, default='./model', help='model save path')

args = parser.parse_args()

deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Don't move this stuff, it needs to be run this early to work


def run():
    """创建模型"""
    model = FFN(in_channels=2, out_channels=1, input_size=args.input_size, delta=args.delta).cuda()

    """数据路径"""
    input_h5data = [args.data]

    """创建data loader"""
    train_dataset = BatchCreator(input_h5data, args.input_size, delta=args.delta, train=True)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=1e-3,  # Learning rate is set by the lr_sched below
        momentum=0.9,
        weight_decay=0.5e-4,
    )

    best_loss = np.inf

    """获取数据流"""

    for iter, (image, targets, seed, coor) in enumerate(train_loader):
        for _, (seeds, images, labels, offsets) in enumerate(
                get_batch(image, targets, seed, args.batch_size, args.input_size, partial(fixed_offsets, fov_moves=train_dataset.shifts))):

            input_data = torch.cat([images, seeds], dim=1)

            input_data = Variable(input_data.cuda())
            seeds = seeds.cuda()
            labels = labels.cuda()

            logits = model(input_data)

            updated = seeds + logits
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(updated, labels)
            loss.backward()
            """梯度截断"""
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_thr)

            optimizer.step()

            diff = (updated.sigmoid()-labels).detach().cpu().numpy()
            accuracy = 1.0*(diff < 0.001).sum() / np.prod(labels.shape)
            print("loss: {}, iteration: {}, Accuracy: {:.2f}%, offset:{}".format(loss.item(), iter, accuracy.item()*100, offsets))

            update_seed(updated, seed, model, offsets)

            """根据最佳loss并且保存模型"""
            if best_loss > loss.item():
                best_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(args.save_path, 'ffn.pth'))
                print('Model saved!')


if __name__ == "__main__":
    seed = int(time.time())
    random.seed(seed)

    run()
