import ot
import ot.plot

# from PIL import Image
# from google.colab.patches import cv2_imshow
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
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


num_sam = 1
# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-512')
boundary = np.load('boundary/biggan/hashiqi/dog.npy')
# Prepare a input
truncation = 0.4
# class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=3)
# noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)
class_vector = one_hot_from_names(['dog'], batch_size=1)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1, seed=12)
class_vector = torch.from_numpy(class_vector)
for i in range(1):
    # Generate visualization pages.
    start_distance = -10
    end_distance = 10
    step = 21
    # np.linspace用来创建等差数列
    distances = np.linspace(start_distance, end_distance, step)
    s = range(10)

    num_sem = 1
    viz_size = 256
    vizer_1 = HtmlPageVisualizer(num_rows=num_sem * (num_sam + 1),
                                 num_cols=step + 1,
                                 viz_size=viz_size)
    vizer_2 = HtmlPageVisualizer(num_rows=num_sam * (num_sem + 1),
                                 num_cols=step + 1,
                                 viz_size=viz_size)

    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer_1.set_headers(headers)
    vizer_2.set_headers(headers)
    for sem_id in range(num_sem):
        value = 1
        vizer_1.set_cell(sem_id * (num_sam + 1), 0,
                         text=f'Semantic {sem_id:03d}<br>({value:.3f})',
                         highlight=True)
        for sam_id in range(num_sam):
            vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, 0,
                             text=f'Sample {sam_id:03d}')
    for sam_id in range(num_sam):
        vizer_2.set_cell(sam_id * (num_sem + 1), 0,
                         text=f'Sample {sam_id:03d}',
                         highlight=True)
        for sem_id in range(num_sem):
            value = 1
            vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, 0,
                             text=f'Semantic {sem_id:03d}<br>({value:.3f})')

    for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):

        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundary[0:1]
            for col_id, d in enumerate(distances, start=1):
                noise_vector_temp = noise_vector.copy()
                noise_vector_temp = np.float32(noise_vector_temp + (d * boundary))
                # All in tensors
                noise_vector_temp = torch.from_numpy(noise_vector_temp)


                with torch.no_grad():
                    output = model(noise_vector_temp, class_vector, truncation)


                image = postprocess(output)[0]
                vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, col_id,
                                 image=image)
                vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, col_id,
                                 image=image)
    model_name = 'BigGAN'
    prefix = (f'{model_name}_'
              f'N{num_sam}_K{num_sem}_seed{i}')
    save_dir = "save_image/cpu_gen/biggan/hashiqi"
    vizer_1.save(os.path.join(save_dir, f'{prefix}_sample_rpca_first.html'))
