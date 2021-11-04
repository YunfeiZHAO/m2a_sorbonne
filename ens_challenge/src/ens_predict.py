import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

from datasets.satellite import SatelliteDataset
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

from utils.utils import show_mask, show_image


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')

    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def ratio(mask):
    batch_size = mask.shape[0]
    size = mask.shape[1]
    res = torch.zeros((batch_size, 10))
    for i in range(10):
        res[:, i] = (mask == i).sum(dim=(1, 2))/(size*size)
    return res


def write_csv(index, matrix, path):
    #index must be a tensor with size = batch_size
    #matrix must be a tensor withe shape =batch_size * 256 * 256
    batch_size = matrix.shape[0]
    idx = index.reshape(batch_size, -1)
    myratio = ratio(matrix)
    data_pure = torch.cat((idx, myratio), dim=1)
    data = pd.DataFrame(data_pure.numpy() , columns = ['sample_id','no_data','clouds','artificial','cultivated','broadleaf','coniferous','herbaceous','natural','snow','water'])
    data = data.astype({'sample_id': 'int'})
    data.to_csv(path, index=False)
    print("make cvs done, with batch size = ", batch_size)


def evaluate(dataset, weight, batch_size, tensorboard_dir, result_save_path, val_csv_path=None, new_csv_truth_path=None):
    args = get_args()
    args.model = weight

    with torch.no_grad():
        # load model
        net = UNet(n_channels=4, n_classes=10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Loading model {args.model}')
        logging.info(f'Using device {device}')
        net.to(device=device)
        net.load_state_dict(torch.load(args.model, map_location=device))
        logging.info('Model loaded!')


        # Create data loaders
        loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
        data_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)
        # (Initialize tensor board logging)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        final_masks = torch.tensor([])
        all_image_ids = torch.tensor([])
        with tqdm(total=dataset.__len__(), desc=f'Testing dataset', unit='img') as pbar:
            for batch in data_loader:
                images = batch['image'].to(device=device)
                image_ids = batch['image_id']
                masks_pred = net(images).float().cpu()
                final_mask = torch.argmax(masks_pred, dim=1).float().cpu()
                i = 0
                # image: 4, H, W
                writer.add_image(f'Prediction_sample-{image_ids[i]}.tif/image',
                                 show_image(images[0]), 0,
                                 dataformats='CHW')

                # masks_pred: C=1, H, W
                writer.add_image(f'Prediction_sample-{image_ids[i]}.tif/predicted_mask',
                                 show_mask(final_mask[0]), 0, dataformats='HWC')

                final_masks = torch.cat((final_masks, final_mask))
                all_image_ids = torch.cat((all_image_ids, image_ids))
                pbar.update(images.shape[0])
        if val_csv_path and new_csv_truth_path:
            val_data = pd.read_csv(val_csv_path)
            val_truth = val_data.loc[val_data['sample_id'].isin(all_image_ids)]
            val_truth.to_csv(new_csv_truth_path, index=False)

        write_csv(all_image_ids, final_masks, result_save_path)



if __name__ == '__main__':
    # 1. test
    test_dataset = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/test_images.csv',
                               images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                               type='test')
    evaluate(dataset=test_dataset,
             weight='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/Unet_experiments2/checkpoint_epoch68.pth',
             batch_size=8,
             tensorboard_dir='../test_experiments2',
             result_save_path='../test_experiments2/test_predicted.csv')

    # 2. validation
    # val_percent = 0.3
    # dataset = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/train_images.csv',
    #                            images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
    #                            type='val')
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    #
    # evaluate(dataset=val_set,
    #          weight='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/Unet_experiments2/checkpoint_epoch68.pth',
    #          batch_size=8,
    #          tensorboard_dir='../val_experiments',
    #          result_save_path='../val_experiments/val_predict.csv',
    #          val_csv_path='../train_labels.csv',
    #          new_csv_truth_path='../val_experiments/val.csv')
