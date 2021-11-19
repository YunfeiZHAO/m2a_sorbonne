from dpt.models import DPTSegmentationModel

import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np
from random import randrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.utils import show_mask, show_image

from datasets.satellite import SatelliteDataset
from datasets.satellite import LandCoverData as LCD

from evaluate import evaluate


def train_net(net,
              device,
              writer,  # tensorboard writer
              dir_checkpoint,
              start_epoch,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              ):
    # 1. Create dataset
    train_set = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/train_index.csv',
                               images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                               type='train')

    val_set = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/val_index.csv',
                               images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                               type='val')
    n_train = int(len(train_set))
    n_val = int(len(val_set))
    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Initialise wandb logging
    # experiment = wandb.init(project='U-Net-ENS', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    # class weights
    # compute class weights for the loss: inverse-frequency balanced
    # note: we set to 0 the weights for the classes "no_data"(0) and "clouds"(1) to ignore these
    class_weight = (1 / LCD.TRAIN_CLASS_COUNTS[2:]) * LCD.TRAIN_CLASS_COUNTS[2:].sum() / (LCD.N_CLASSES-2)
    class_weight = torch.tensor(np.append(np.zeros(2), class_weight), dtype=torch.float)
    logging.info(f"Will use class weights: {class_weight}, type: {type(class_weight)}")
    criterion_mask = nn.CrossEntropyLoss(weight=class_weight).to(device)
    criterion_kl = torch.nn.KLDivLoss()

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, true_ratios = batch['image'], batch['mask'], batch['label']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)  # B, h, w
                true_ratios = true_ratios.to(device=device, dtype=torch.float32)  # B, 10
                # Loss

                with torch.cuda.amp.autocast(enabled=amp):
                    # predict mask and calculate ratio

                    masks_pred = net(images)  # B, C, h, w
                    # writer.add_graph(net, images)

                    batch_ratio = torch.sum(masks_pred, (-2, -1))  # B, 10
                    batch_ratio /= torch.sum(batch_ratio, 1)[..., None]
                    # generate loss for ratios and masks
                    loss_mask = criterion_mask(masks_pred, true_masks)
                    kl_loss_ratio = criterion_kl(batch_ratio.softmax(dim=-1).log(), true_ratios)
                    loss = loss_mask + kl_loss_ratio * 10

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('Train batch/mask loss', loss_mask.item(), global_step)
                writer.add_scalar('Train batch/ratio kl loss', kl_loss_ratio.item(), global_step)
                writer.add_scalar('Train batch/total loss', loss.item(), global_step)

                # Evaluation round
                with torch.no_grad():
                    division_step = 60 * batch_size
                    # division_step = 1  # for debugging
                    if division_step > 0:
                        if global_step % division_step == 0:
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                writer.add_histogram('Weights/' + tag, value.data.cpu(), start_epoch+epoch)
                                writer.add_histogram('Gradients/' + tag, value.grad.data.cpu(), start_epoch+epoch)

                            val_crossentropy_loss_mask, val_crossentropy_loss_ratio, kl_loss = evaluate(net, val_loader, device)
                            # for debug
                            # val_score = torch.tensor(0.3081, device='cuda:0')
                            scheduler.step()

                            logging.info('Validation kl_loss: {}'.format(kl_loss))
                            writer.add_scalar('Evaluation/learning rate', optimizer.param_groups[0]['lr'], global_step)
                            writer.add_scalar('Evaluation/validation mask crossentropy loss',
                                              val_crossentropy_loss_mask, global_step)
                            writer.add_scalar('Evaluation/validation ratio crossentropy loss',
                                              val_crossentropy_loss_ratio, global_step)
                            writer.add_scalar('Evaluation/validation kl score', kl_loss, global_step)

                            # For this training batch
                            i = randrange(len(batch['image_id']))
                            image_id = batch['image_id'][i]
                            # image: B, 4, H, W
                            writer.add_image(f'Evaluation-global_step{global_step}/{image_id}-image', show_image(images[i]).float().cpu(), 0,
                                             dataformats='CHW')
                            # true_masks: B, H, W
                            writer.add_image(f'Evaluation-global_step{global_step}/{image_id}-true_mask', show_mask(true_masks[i]).float().cpu(), 0,
                                             dataformats='HWC')
                            # masks_pred: B, C=10, H, W
                            writer.add_image(f'Evaluation-global_step{global_step}/{image_id}-predicted_mask',
                                             show_mask(torch.argmax(masks_pred[i], dim=0)).float().cpu(), 0, dataformats='HWC')

                            # For random image in validation set
                            j = randrange(n_val)
                            val_sample = val_set.__getitem__(j)
                            image_id = val_sample['image_id']
                            image = val_sample['image'].unsqueeze(dim=0).to(device=device, dtype=torch.float32)
                            true_mask = val_sample['mask']
                            masks_pred = net(image)
                            # image: 4, H, W
                            writer.add_image(f'Evaluation-global_step{global_step}/{image_id}-val_image', show_image(image[0]).float().cpu(), 0,
                                             dataformats='CHW')
                            # true_masks: H, W
                            writer.add_image(f'Evaluation-global_step{global_step}/{image_id}-val_true_mask', show_mask(true_mask), 0,
                                             dataformats='HWC')
                            # masks_pred: C=10, H, W
                            writer.add_image(f'Evaluation-global_step{global_step}/{image_id}-val_predicted_mask',
                                             show_mask(torch.argmax(masks_pred[0], dim=0)).float().cpu(), 0, dataformats='HWC')

        writer.add_scalar('Train/Epoch_loss', epoch_loss, start_epoch+epoch)

        if save_checkpoint:
            torch.save({
                'epoch': start_epoch+epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(dir_checkpoint, 'checkpoint_epoch{}.pth'.format(start_epoch+epoch)))
            logging.info(f'Checkpoint {start_epoch+epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    root = '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/'
    run_name = 'test6'
    # hyper parameters
    args = get_args()
    args.epochs = 100
    args.batch_size = 8
    args.lr = 0.001

    # args.load = '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/test3/checkpoint_epoch20.pth'

    # for tensorboard visualisation
    args.tensorboard_dir = os.path.join(root, Path(f'experiments/{run_name}'))
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    # for saving checkpoints
    args.dir_checkpoint = os.path.join(root, Path(f'checkpoints/{run_name}'))
    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
    # (Initialize tensor board logging)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # Use DPT
    net = DPTSegmentationModel(
        150,
        path=None,
        backbone="vitb_rn50_384",
    )

    logging.info(f'Network: DPT, dpt_hybrid\n')

    start_epoch = 1
    # if args.load:
    #     checkpoint = torch.load(args.load, map_location=device)
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  writer=writer,
                  dir_checkpoint=args.dir_checkpoint,
                  start_epoch=start_epoch,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
