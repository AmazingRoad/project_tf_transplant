import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='./data_raw4_focus_500_filter1.5.h5', help='input images')
parser.add_argument('--label', type=str, default='./pred.h5', help='input images')
parser.add_argument('--model', type=str, default='./model/ffn.pth', help='path to ffn model')
parser.add_argument('--delta', default=(12, 12, 12), help='delta offset')
parser.add_argument('--input_size', default=(51, 51, 51), help='input size')
parser.add_argument('--depth', type=int, default=20, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.9, help='input size')
parser.add_argument('--act_thr', type=float, default=0.95, help='input size')

args = parser.parse_args()


def run():
    """创建模型"""
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)

    """载入模型"""
    model.load_state_dict(torch.load(args.model))
    model.eval()

    """读取数据"""
    with h5py.File(args.data, 'r') as f:
        images = (f['image'][()].astype(np.float32) - 128) / 33

    """创建分割实例"""
    canva = Canvas(model, images, args.input_size, args.delta, args.seg_thr, args.mov_thr, args.act_thr)
    """开始分割"""
    canva.segment_all()
    canva.segmentation[canva.segmentation < 0] = 0
    # Save segmentation results. Reduce # of bits per item if possible.
    save_subvolume(
        canva.segmentation,
        unalign_origins(canva.origins, np.array((0, 0, 0))),
        'result/pred',
        overlaps=canva.overlaps)
    id, count = np.unique(canva.segmentation, return_counts=True)
    with h5py.File('pred.h5', 'w') as g:
        g.create_dataset('label', data=canva.segmentation, compression='gzip')
    # with h5py.File('pred_all.h5', 'w') as g:
    #     for idx in range(len(canva.target_dic)):
    #         g.create_dataset('label_{}'.format(idx+1), data=canva.target_dic[idx+1], compression='gzip')
    print("id:{}, count:{}".format(id, count))


if __name__ == '__main__':
    run()
