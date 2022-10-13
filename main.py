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
from RobustPCA import RobustPCA
from torch import nn
import pandas as pd
import torchvision.transforms as transforms

# import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_idx = 'all'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = 'stylegan_celebahq1024'
num_sam = 1

generator = load_generator(model_name)
gan_type = parse_gan_type(generator)

def main():
    np.random.seed(4)
    torch.manual_seed(4)
    codes = torch.randn(num_sam, generator.z_space_dim).to(device)


    codes = generator.mapping(codes)['w']
    codes = codes.detach().cpu().numpy()

    codes_trun = codes.copy()


    codes = generator.truncation(to_tensor(codes),
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    image1 = generator.synthesis(codes)['image']
    image_show = postprocess(image1)[0]
    # plt.imshow(image_show)
    # plt.show()

    image_w = torch.mean(to_tensor(image_show), axis=2)
    # t = transforms.Compose([
    #     transforms.Resize((512, 512)), ])
    # image_w = t(image_w)



    se = pd.Series(codes_trun.ravel())
    w_proportitionDict = dict(se.value_counts(normalize=True))
    w_vol = np.array(list(w_proportitionDict.keys()))
    w_propor = np.array(list(w_proportitionDict.values()))

    img0 = np.round(image_w.detach().cpu().numpy())
    img_se = pd.Series(img0.reshape(1, -1)[0])
    img_proportitionDict = dict(img_se.value_counts(normalize=True))
    img_vol = np.array(list(img_proportitionDict.keys()))
    img_propor = np.array(list(img_proportitionDict.values()))
    # plt.imshow(img0)
    # plt.show()
    img_vol = img_vol.reshape(len(img_vol), 1)
    img_vol = img_vol / 255


    M = ot.dist(w_vol.reshape(len(w_vol), 1), img_vol)
    M /= M.max()


    T_img = ot.sinkhorn(w_propor, img_propor, M, 1)


    coef_f = 1 / T_img.shape[0]
    T_img_m = coef_f * T_img.dot(T_img.T)
    # T_img_m = coef_f * T_img
    RPCA = RobustPCA(T_img_m, lamb=1 / 60)
    L_f, _ = RPCA.fit(max_iter=10000)
    L_f = L_f / np.linalg.norm(L_f, axis=0, keepdims=True)
    s, u = np.linalg.eig(L_f)
    # u, s, v = np.linalg.svd(L_f)

    boundaries_1 = u.astype(np.float32)

    # Generate visualization pages.
    start_distance = -5.0
    end_distance = 5.0
    step = 21
    # np.linspace用来创建等差数列
    distances = np.linspace(start_distance, end_distance, step)

    num_sem = 20
    viz_size = 256
    vizer_1 = HtmlPageVisualizer(num_rows=num_sem * (num_sam + 1),
                                 num_cols=step + 1,
                                 viz_size=viz_size)


    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer_1.set_headers(headers)
    for sem_id in range(num_sem):
        value = s[sem_id]
        vizer_1.set_cell(sem_id * (num_sam + 1), 0,
                         text=f'Semantic {sem_id:03d}<br>({value:.3f})',
                         highlight=True)
        for sam_id in range(num_sam):
            vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, 0,
                             text=f'Sample {sam_id:03d}')

    code = codes_trun
    for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
        boundary = boundaries_1[sem_id:sem_id + 1]
        for col_id, d in enumerate(distances, start=1):
            temp_code = code.copy()
            temp_code += boundary * d
            temp_code = generator.truncation(to_tensor(temp_code),
                                         trunc_psi=0.7,
                                         trunc_layers=8)
            image = generator.synthesis(temp_code)['image']
            image = postprocess(image)[0]
            vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, col_id,
                             image=image)




    prefix = (f'{model_name}_'
              f'N{num_sam}_K{num_sem}_seed{4}')
    save_dir = "./result/stylegan_ffhq"
    vizer_1.save(os.path.join(save_dir, f'{prefix}_sample_rpca_first.html'))


if __name__ == '__main__':
    main()
