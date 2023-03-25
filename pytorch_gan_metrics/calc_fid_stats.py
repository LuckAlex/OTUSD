import argparse
import os

import numpy as np
from torch.utils.data import DataLoader

from utils import ImageDataset
from core import get_inception_feature


def calc_and_save_stats(path, output, batch_size):
    dataset = ImageDataset(path, exts=['png', 'jpg'])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    acts, = get_inception_feature(loader, dims=[2048], verbose=True)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    if os.path.dirname(output) != "":
        os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(output, mu=mu, sigma=sigma)


calc_and_save_stats(path='./data/ffhq_smile_256', output='./ffhq_smile', batch_size=50)

