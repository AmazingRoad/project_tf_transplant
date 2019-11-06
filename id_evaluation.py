import h5py
import numpy as np
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='script to evaluate id recall')
parser.add_argument('--gt', type=str, default='./data.h5', help='ground truth data')
parser.add_argument('--pred', type=str, default='./pred.h5', help='prediction h5 file')

args = parser.parse_args()


def run():
    with h5py.File(args.gt, 'r') as f:
        label = f['label'][()]

    id_L, countL = np.unique(label, return_counts=True)

    with h5py.File(args.pred, 'r') as f:
        pred = f['label'][()]

    mergercnt = 0
    noncnt = 0
    consize = 0
    for id in id_L:

        print("********************")
        print("id in GT", id)
        maskid_L = (label == id)
        maskid_Lsize = np.sum(maskid_L)
        print("id vol", np.sum(maskid_L))

        idInPred = stats.mode(pred[maskid_L])
        idInPredNum = idInPred[0][0]
        print("idInPred", idInPredNum)

        maskcapnum = (pred[maskid_L] == idInPredNum)

        print("idInPred vol", np.sum(maskcapnum))
        print(np.sum(maskcapnum) / maskid_Lsize)

        maskid_P = (pred == idInPredNum)
        maskid_Psize = np.sum(maskid_P)

        if idInPred[0][0] == 0:
            print("non")
            noncnt += 1

        elif maskid_Psize >= maskid_Lsize * 1.5:
            print("merger")
            mergercnt += 1
        else:
            if np.sum(maskcapnum) / maskid_Lsize >= 0.6:
                consize += maskid_Lsize

    print("non", noncnt)
    print("merger:", mergercnt)
    print("con", consize)
    masknum = (label > 0)
    print(np.sum(masknum))
    print(consize / np.sum(masknum))
    maskZ = (pred == 0)
    print(np.sum(maskZ))


if __name__ == "__main__":
    run()
