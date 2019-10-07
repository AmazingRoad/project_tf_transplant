import itertools
from scipy.special import logit
import torch
import six
import numpy as np


def make_seed(shape, pad=0.05, seed=0.95):
    """创建种子"""
    seed_array = np.full(list(shape), pad, dtype=np.float32)
    idx = tuple([slice(None)] + list(np.array(shape) // 2))
    seed_array[idx] = seed
    return seed_array


def fixed_offsets(seed, fov_moves, threshold=0.9):
    """offset偏移."""
    for off in itertools.chain([(0, 0, 0)], fov_moves):
        is_valid_move = seed[0,
                            seed.shape[1] // 2 + off[2],
                            seed.shape[2] // 2 + off[1],
                            seed.shape[3] // 2 + off[0]
                        ] >= logit(np.array(threshold))

        if not is_valid_move:
            continue

        yield off


def center_crop_and_pad(data, coor, target_shape):
    """根据中心坐标 crop patch"""
    target_shape = np.array(target_shape)

    start = coor - target_shape // 2
    end = start + target_shape

    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    cropped = data[tuple(selector)]

    if target_shape is not None:
        target_shape = np.array(target_shape)
        delta = target_shape - cropped.shape
        pre = delta // 2
        post = delta - delta // 2

        paddings = []  # no padding for batch
        paddings.extend(zip(pre, post))

        cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def crop_and_pad(data, offset, crop_shape, target_shape=None):
    """根据offset crop patch"""
    # Spatial dimensions only. All vars in zyx.
    shape = np.array(data.shape[1:])
    crop_shape = np.array(crop_shape)
    offset = np.array(offset[::-1])

    start = shape // 2 - crop_shape // 2 + offset
    end = start + crop_shape

    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + selector)
    cropped = data[selector]

    if target_shape is not None:
        target_shape = np.array(target_shape)
        delta = target_shape - crop_shape
        pre = delta // 2
        post = delta - delta // 2

        paddings = [(0, 0)]  # no padding for batch
        paddings.extend(zip(pre, post))
        paddings.append((0, 0))  # no padding for channels

        cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def get_example(loader, shape, get_offsets):

    while True:
        for iter, (image, targets, seed, coor) in enumerate(loader):
            for off in get_offsets(seed):
                predicted = crop_and_pad(seed, off, shape).unsqueeze(0)
                patches = crop_and_pad(image, off, shape).unsqueeze(0)
                labels = crop_and_pad(targets, off, shape).unsqueeze(0)
                offset = off

                yield predicted, patches, labels, offset


def get_batch(loader, batch_size, shape, get_offsets):
    def _batch(iterable):
        for batch_vals in iterable:
          yield zip(*batch_vals)

    for seeds, patches, labels, offsets in _batch(six.moves.zip(
        *[get_example(loader, shape, get_offsets) for _
            in range(batch_size)])):

        yield torch.cat(seeds, dim=0).float(), torch.cat(patches, dim=0).float(), \
              torch.cat(labels, dim=0).float(), offsets


def update_seed(updated, seed, model, offsets):
    for idx, offset in enumerate(offsets):
        start = offset + model.radii - model.input_size // 2
        end = start + model.input_size
        assert np.all(start >= 0)

        selector = [slice(s, e) for s, e in zip(start, end)]
        seed[0][selector] = torch.squeeze(updated[idx]).detach().cpu()
