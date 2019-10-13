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
import h5py


parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).')
parser.add_argument('-d', '--data', type=str, default='./data1.h5', help='training data')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--delta', default=(5, 5, 5), help='delta offset')
parser.add_argument('--input_size', default=(31, 31, 31), help='input size')
parser.add_argument('--clip_grad_thr', type=float, default=0.7, help='grad clip threshold')
parser.add_argument('--save_path', type=str, default='./model', help='model save path')
parser.add_argument('--resume', type=str, default=None, help='resume training')
parser.add_argument('--interval', type=int, default=120, help='How often to save model (in seconds).')
parser.add_argument('--iter', type=int, default=1e100, help='training iteration')

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
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    best_loss = np.inf

    """获取数据流"""
    t_last = time.time()
    cnt = 0
    tp = fp = tn = fn = 0
    batch_it = get_batch(train_loader, args.batch_size, args.input_size,
                         partial(fixed_offsets, fov_moves=train_dataset.shifts))

    while cnt < args.iter:
        cnt += 1
        # for iter, (seeds, images, labels, offsets) in enumerate(
        #         get_batch(train_loader, args.batch_size, args.input_size,
        #                   partial(fixed_offsets, fov_moves=train_dataset.shifts))):

        seeds, images, labels, offsets = next(batch_it)

        t_curr = time.time()
        """正样本权重"""
        pos_w = - torch.log(1e-3 + (labels > 0.5).sum().float() / np.prod(labels.shape))
        slice = sigmoid(seeds[:, :, seeds.shape[2] // 2, :, :])
        seeds[:, :, seeds.shape[2] // 2, :, :] = slice
        labels = labels.cuda()

        torch_seed = torch.from_numpy(seeds)
        input_data = torch.cat([images, torch_seed], dim=1)
        input_data = Variable(input_data.cuda())

        logits = model(input_data)
        updated = torch_seed.cuda() + logits

        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(updated, labels, pos_weight=pos_w)
        loss.backward()
        """梯度截断"""
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_thr)
        optimizer.step()

        seeds[...] = updated.detach().cpu().numpy()

        pred_mask = (updated >= logit(0.9)).detach().cpu().numpy()
        true_mask = (labels > 0.5).cpu().numpy()
        true_bg = np.logical_not(true_mask)
        pred_bg = np.logical_not(pred_mask)
        tp += (true_mask & pred_mask).sum()
        fp += (true_bg & pred_mask).sum()
        fn += (true_mask & pred_bg).sum()
        tn += (true_bg & pred_bg).sum()
        precision = 1.0 * tp / max(tp + fp, 1)
        recall = 1.0 * tp / max(tp + fn, 1)
        accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
        print('[Iter_{}: offset: {}, loss: {:.4}, Precision: {:.2f}%, Recall: {:.2f}%, '
                         'Accuracy: {:.2f}%]\r'.format(cnt, offsets, loss.item(), precision*100,
                                                       recall*100, accuracy * 100))

        """根据最佳loss并且保存模型"""
        if best_loss > loss.item() or t_curr - t_last > args.interval:
            tp = fp = tn = fn = 0
            t_last = t_curr
            best_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(args.save_path, 'ffn.pth'))
            print('Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%, Model saved!'.format(
                precision * 100, recall * 100, accuracy * 100))
            im = images[0, 0, :, :, :] * 33 + 128
            label = (labels[0, 0, :, :, :] > 0.5)
            pred = (updated.sigmoid()[0, 0, :, :, :] > 0.5)
            im = im.detach().cpu().numpy().astype(np.uint8)
            label = label.detach().cpu().numpy().astype(np.uint8) * 255
            pred = pred.detach().cpu().numpy().astype(np.uint8) * 255
            with h5py.File('data/sample_{}.h5'.format(cnt), 'w') as f:
                f.create_dataset('image', data=im, compression='gzip')
                f.create_dataset('label', data=label, compression='gzip')
                f.create_dataset('pred', data=pred, compression='gzip')


if __name__ == "__main__":
    seed = int(time.time())
    random.seed(seed)

    run()
