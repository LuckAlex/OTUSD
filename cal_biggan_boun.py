import torch
import numpy as np
import pandas as pd
from RobustPCA import RobustPCA
import ot
import ot.plot
from utils import postprocess
import matplotlib.pyplot as plt
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-512')

# Prepare a input
truncation = 0.4
# class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=3)
# noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)
class_vector = one_hot_from_names(['dog'], batch_size=1)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1, seed=12)


# All in tensors
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)


# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)
image = postprocess(output)

plt.imshow(image[0])
plt.show()

se = pd.Series(noise_vector[0])
w_proportitionDict = dict(se.value_counts(normalize=True))
w_vol = np.array(list(w_proportitionDict.keys()))
w_propor = np.array(list(w_proportitionDict.values()))


img_se = pd.Series(output.reshape(1, -1)[0])
img_proportitionDict = dict(img_se.value_counts(normalize=True))
img_vol = np.array(list(img_proportitionDict.keys()))
img_propor = np.array(list(img_proportitionDict.values()))

M = ot.dist(w_vol.reshape(len(w_vol), 1), img_vol.reshape(len(img_vol), 1))
M /= M.max()

T_img = ot.sinkhorn(w_propor, img_propor, M, 1)

# eigen_values_1, eigen_vectors_1 = np.linalg.eig(T_img1.dot(T_img1.T))
# eigen_values_2, eigen_vectors_2 = np.linalg.eig(T_img2.dot(T_img2.T))
coef_f = 1 / T_img.shape[0]
T_img_m = coef_f * T_img.dot(T_img.T)
RPCA = RobustPCA(T_img_m, lamb=1 / 60)
L_f, _ = RPCA.fit(max_iter=10000)
u, s, v = np.linalg.svd(L_f)
# u, s, v = np.linalg.svd(T_img.dot(T_img.T))

boundaries_1 = u
np.save('boundary/biggan/hashiqi/dog.npy', boundaries_1)
