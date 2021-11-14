import argparse
import logging
from tqdm import tqdm

import pandas as pd

import torch
from unet import UNet
from datasets.satellite import SatelliteDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import show_mask, show_image, write_csv


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


@torch.no_grad()
def evaluate(dataset, weight, batch_size, tensorboard_dir, result_save_path, write_image=False, val_csv_path=None, new_csv_truth_path=None):
    args = get_args()
    args.model = weight

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
    if write_image:
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
            if write_image:
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
    # test_dataset = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/test_images.csv',
    #                                 images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
    #                                 type='test')
    # evaluate(dataset=test_dataset,
    #          weight='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/test2/checkpoint_epoch7.pth',
    #          batch_size=8,
    #          tensorboard_dir='../test_experiments4',
    #          result_save_path='../test_experiments4/test_predicted.csv')

    # 2. validation
    val_set = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/val_index.csv',
                               images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                               type='val')
    #
    evaluate(dataset=val_set,
             weight='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/test/checkpoint_epoch7.pth',
             batch_size=8,
             tensorboard_dir='../val_experiments',
             result_save_path='../val_experiments/val_predict.csv',
             val_csv_path='../train_labels.csv',
             new_csv_truth_path='../val_experiments/val.csv')
