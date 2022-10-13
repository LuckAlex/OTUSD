import ot
import ot.plot

# from PIL import Image
# from google.colab.patches import cv2_imshow

import os
import argparse
from tqdm import tqdm
import numpy as np
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import HtmlPageVisualizer
from utils import parse_indices
#from utils import dou2unit
from RobustPCA import RobustPCA
from torch import nn
import pandas as pd
import torchvision.transforms as transforms

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_idx = 'all'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = 'stylegan2_ffhq1024'
output_dir = 'H:\\otsd'
num = 30000
pbar = tqdm(total=num, leave=False)

generator = load_generator(model_name)
gan_type = parse_gan_type(generator)

width = 1024
height = 1024 # keep original height
dim = (width, height)


for seed in range(num):
    np.random.seed(seed)
    torch.manual_seed(seed)
    codes = torch.randn(1, generator.z_space_dim)
    # codes = torch.Tensor(np.random.randn(1, generator.z_space_dim)).to(device)
    # codes = np.load('./noise/gender/4.npy')
    codes = torch.Tensor(codes).to(device)
    codes = generator.mapping(codes)['w']
    codes = generator.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    codes = codes.detach().cpu().numpy()






    u = np.load('boundary/stylegan2_ffhq/smile/stylegan2_ffhq_smile.npy')


    temp_code = codes.copy()
    a = u[:, 0:512]
    b = u[:, 512:1024]
    temp_code[:, 0:8, :] += a * (5)
    temp_code[:, 8:, :] += b * (5)

    image = generator.synthesis(to_tensor(temp_code))['image']
    image = postprocess(image)[0]
    resized = cv2.resize(image[:, :, ::-1], dim, interpolation=cv2.INTER_AREA)
    save_path = os.path.join(output_dir, f'{pbar.n:06d}.png')
    cv2.imwrite(save_path, resized)
    pbar.update(1)


