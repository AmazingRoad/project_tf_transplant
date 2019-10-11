import os
import sys
import numpy as np
import time
from scipy.special import expit
from scipy.special import logit
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='./images.h5', help='input images')
parser.add_argument('--label', type=str, default='./groundtruth.h5', help='input images')
parser.add_argument('--model', type=str, default='./model/ffn1.pth', help='path to ffn model')
parser.add_argument('--delta', default=(8, 8, 8), help='delta offset')
parser.add_argument('--input_size', default=(33, 33, 33), help='input size')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.9, help='input size')
parser.add_argument('--act_thr', type=float, default=0.95, help='input size')

args = parser.parse_args()


def run():
    model = FFN(in_channels=2, out_channels=1, input_size=args.input_size, delta=args.delta).cuda()

    assert os.path.isfile(args.model)

    model.load_state_dict(torch.load(args.model))

    with h5py.File(args.data, 'r') as f:
        images = (f['raw'].value.astype(np.float32) - 128) / 33
        # labels = g['label'].value

    canva = Canvas(model, images, args.input_size, args.delta, args.seg_thr, args.mov_thr, args.act_thr)
    canva.segment_all()
    with h5py.File(args.label, 'w') as g:
        f.create_dataset('raw', data=canva.segmentation, compression='gzip')
    print('done')


if __name__ == '__main__':
    run()
