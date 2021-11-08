import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from unet import UNet
from datasets.satellite import SatelliteDataset

from torch.utils.tensorboard import SummaryWriter


def evaluate(net, dataloader, device):
    """
    Calculate dice score, cross_entropy, KL loss
    """

    net.eval()
    batch_size = dataloader.batch_size
    num_val_batches = len(dataloader)
    dice_score = 0
    val_crossentropy_loss = 0
    KL_score = 0
    crossentropy = torch.nn.CrossEntropyLoss().to(device)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image_ids, image, mask_true, ratio_label = batch['image_id'], batch['image'], batch['mask'], batch['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            final_mask = torch.argmax(mask_pred, dim=1)
            # KL_score
            predict_ratio = calculate_class_ratio(final_mask)
            predict_ratio /= predict_ratio.sum(dim=1)[..., None]
            ratio_label /= ratio_label.sum(dim=1)[..., None]
            KL_score += torch.sum(ratio_label * np.log((ratio_label + 1e-7) / (predict_ratio + 1e-7)))

            # cross entropy estimation
            val_crossentropy_loss += crossentropy(mask_pred, mask_true).float().cpu()
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true_onehot, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true_onehot[:, 1:, ...], reduce_batch_first=False)
    net.train()

    KL_score = KL_score / (num_val_batches * batch_size)
    dice_score = dice_score / num_val_batches
    return dice_score, val_crossentropy_loss, KL_score


def calculate_class_ratio(mask):
    """get class ratio from a batch of mask, we do not consider the first two class
    :param mask: B, H, W
    """
    batch_size, h, w = mask.size()
    res = torch.zeros((batch_size, 8))
    for i, j in enumerate(np.arange(2, 10)):
        res[:, i] = (mask == j).sum(dim=(1, 2))/(h*w)
    return res


# tensorboard_dir = '../val'
# writer = SummaryWriter(log_dir=tensorboard_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_set = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/val_index.csv',
                           images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                           type='val')
loader_args = dict(batch_size=8, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(n_channels=4, n_classes=10, bilinear=True).to(device)
load = '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/test/checkpoint_epoch6.pth'
net.load_state_dict(torch.load(load, map_location=device))
dice_score, val_crossentropy_loss, KL_score = evaluate(net, val_loader, device)
print(f"dice_score: {dice_score}, val_crossentropy_loss: {val_crossentropy_loss}, KL_score: {KL_score}")
