import argparse
import time
import random
import zipfile
from torch.utils.data import DataLoader
from core.data.utils import *
from functools import partial
import os
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument(
    '-s', '--epoch-size', type=int, default=100,
    help='How many training samples to process between '
         'validation/preview/extended-stat calculation phases.'
)
parser.add_argument(
    '-m', '--max-steps', type=int, default=500000,
    help='Maximum number of training steps to perform.'
)
parser.add_argument(
    '-t', '--max-runtime', type=int, default=3600 * 24 * 4,  # 4 days
    help='Maximum training time (in seconds).'
)
parser.add_argument(
    '-r', '--resume', metavar='PATH',
    help='Path to pretrained model state dict or a compiled and saved '
         'ScriptModule from which to resume training.'
)
parser.add_argument(
    '--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).'
)
parser.add_argument('-i', '--ipython', action='store_true',
    help='Drop into IPython shell on errors or keyboard interrupts.'
)
args = parser.parse_args()

deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

# Don't move this stuff, it needs to be run this early to work
import core
core.select_mpl_backend('Agg')

from core.training import metrics
from core.models.ffn import FFN
from core.data import BatchCreator

device = torch.device('cuda')


def run():
    model = FFN(in_channels=2, out_channels=1).to(device)

    save_root = os.path.expanduser('./log/ffn/')
    os.makedirs(save_root, exist_ok=True)

    input_h5data = ['./data.h5']

    if args.resume is not None:  # Load pretrained network
        pretrained = os.path.expanduser(args.resume)
        _warning_str = 'Loading model without optimizer state. Prefer state dicts'
        if zipfile.is_zipfile(pretrained):  # Zip file indicates saved ScriptModule
            logger.warning(_warning_str)
            model = torch.jit.load(pretrained, map_location=device)
        else:  # Either state dict or pickled model
            state = torch.load(pretrained)
            if isinstance(state, dict):
                model.load_state_dict(state['model_state_dict'])
                optimizer_state_dict = state.get('optimizer_state_dict')
                lr_sched_state_dict = state.get('lr_sched_state_dict')
                if optimizer_state_dict is None:
                    logger.warning('optimizer_state_dict not found.')
                if lr_sched_state_dict is None:
                    logger.warning('lr_sched_state_dict not found.')
            elif isinstance(state, nn.Module):
                logger.warning(_warning_str)
                model = state
            else:
                raise ValueError(f'Can\'t load {pretrained}.')

    train_dataset = BatchCreator(input_h5data, (33, 33, 33), delta=(8, 8, 8), train=True)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,  # Learning rate is set by the lr_sched below
        # momentum=0.9,
        weight_decay=0.5e-4,
    )

    valid_metrics = {
        'val_accuracy': metrics.bin_accuracy,
        'val_precision': metrics.bin_precision,
        'val_recall': metrics.bin_recall,
        'val_DSC': metrics.bin_dice_coefficient,
        'val_IoU': metrics.bin_iou,
    }

    best_loss = np.inf

    # for iter, (data, label, seed, coor)in enumerate(train_loader):

    for _, (seeds, images, labels, offsets) in enumerate(
            get_batch(train_loader, 4, [33, 33, 33], partial(fixed_offsets, fov_moves=train_dataset.shifts))):
        # seeds, images, labels, offsets = get_batch(data, label, seed, 4, [33, 33, 33], partial(fixed_offsets, fov_moves=train_dataset.shifts))

        input_data = torch.cat([images, seeds], dim=1)

        input_data = Variable(input_data.cuda())
        seeds = seeds.cuda()
        labels = labels.cuda()

        logits = model(input_data)

        updated = seeds + logits
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(updated, labels)
        loss.backward()
        optimizer.step()
        # if loss.item() < 1:
        #     for idx in range(len(images)):
        #         im = images[idx, 0, :, :][1, :, :].cpu().numpy().astype(np.uint8)
        #         label = labels[idx, 0, :, :].detach().cpu().numpy()*255
        #         label = label.astype(np.uint8)[1, :, :]
        #         logit = torch.sigmoid(updated)*255
        #         logit = logit[idx, 0, :, :].detach().cpu().numpy().astype(np.uint8)[1, :, :]
        #         cv2.imshow('im', im)
        #         cv2.imshow('logit', logit)
        #         cv2.imshow('label', label)
        #         cv2.waitKey()
        print("loss: {}, offset: {}".format(loss.item(), offsets))
        # update_seed(updated, seeds, model, offsets)

        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save(model.state_dict(), 'test.pth')
            print('Epoch: model is saved')


if __name__ == "__main__":
    seed = int(time.time())
    random.seed(seed)

    run()
