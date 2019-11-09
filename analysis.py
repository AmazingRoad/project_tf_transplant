import h5py
import argparse
import argparse

parser = argparse.ArgumentParser(description='analysis tool')
parser.add_argument('--pred', type=str, default='./pred.h5', help='predict segmentation mask')
parser.add_argument('--gt', type=str, default='./data_raw4_focus_500_filter1.5.h5', help='ground truth label')

args = parser.parse_args()


def run():
    with h5py.File(args.pred, 'r') as f:
        seg = f['label'][()]

    mask_pred = seg > 0

    with h5py.File(args.gt, 'r') as f:
        gt = f['label'][()]

    mask_gt = gt > 0

    capture = (mask_pred*mask_gt).sum() / mask_gt.size

    print("capture: {}".format(capture))


if __name__ == "__main__":
    run()
