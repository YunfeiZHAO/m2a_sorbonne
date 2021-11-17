import argparse
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from unet import UNet
from datasets.satellite import SatelliteDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import show_mask, show_image, write_csv

CSV_FILE_Y_TRUE = '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/experiments/evaluation.csv'
CSV_FILE_Y_PRED = '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/experiments/evaluation_label.csv'

# CSV_FILE_Y_TRUE = './test_experiments2/test_predicted.csv'
# CSV_FILE_Y_PRED = './test_predicted_random.csv'

df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')

df_y_pred = df_y_pred.loc[df_y_true.index]
# remove ignored class columns "no_data" and "clouds" if present
df_y_pred = df_y_pred.drop(['no_data', 'clouds'], axis=1, errors='ignore')
df_y_true = df_y_true.drop(['no_data', 'clouds'], axis=1, errors='ignore')

predict_ratio = torch.tensor(df_y_pred.values, dtype=torch.float64)
true_labels = torch.tensor(df_y_true.values, dtype=torch.float64)
predict_ratio /= predict_ratio.sum(dim=1)[..., None]
true_labels /= true_labels.sum(dim=1)[..., None]
KL_score = torch.mean(torch.sum(true_labels * np.log((true_labels + 1e-7) / (predict_ratio + 1e-7)), 1))
print('score:', KL_score)