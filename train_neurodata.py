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
    '--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).'
)

parser.add_argument('-d', '--data', type=str, default='./data.h5', help='training data')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--delta', default=(8, 8, 8), help='delta offset')
parser.add_argument('--input_size', default=(33, 33, 33), help='input size')
parser.add_argument('--clip_grad_thr', type=float, default=0.7, help='grad clip threshold')
parser.add_argument('--save_path', type=str, default='./model', help='model save path')
parser.add_argument('--resume', type=str, default=None, help='resume training')

args = parser.parse_args()

deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def run():
    """创建模型"""
    model = FFN(in_channels=2, out_channels=1, input_size=args.input_size, delta=args.delta).cuda()

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    """数据路径"""
    input_h5data = [args.data]

    """创建data loader"""
    train_dataset = BatchCreator(input_h5data, args.input_size, delta=args.delta, train=True)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

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
            loss = F.binary_cross_entropy(updated.sigmoid(), labels)
            loss.backward()
            """梯度截断"""
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_thr)
            optimizer.step()

            update_seed(updated, seed, model, offsets)

            pred_mask = (updated.sigmoid() > logit(0.9)).detach().cpu().numpy()
            true_mask = (labels > 0.5).cpu().numpy()
            true_bg = np.logical_not(true_mask)
            pred_bg = np.logical_not(pred_mask)
            tp = (true_mask * pred_mask).sum()
            fp = (true_bg * pred_mask).sum()
            fn = (true_mask * pred_bg).sum()
            tn = (true_bg * pred_bg).sum()
            precision = 1.0 * tp / max(tp + fp, 1)
            recall = 1.0 * tp / max(tp + fn, 1)
            accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
            print("loss: {:.2f}, iteration: {}, Accuracy: {:.2f}%, offset:{}".format(
                loss.item(), iter, accuracy * 100, offsets))

            """根据最佳loss并且保存模型"""
            if best_loss > loss.item():
                best_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(args.save_path, 'ffn.pth'))
                print('Model saved!')


if __name__ == "__main__":
    seed = int(time.time())
    random.seed(seed)

    run()
