import argparse
import logging
import sys
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

from utils.dice_score import dice_loss
from utils.utils import show_mask, show_image

from evaluate import evaluate
from unet import UNet

from datasets.satellite import SatelliteDataset
from datasets.satellite import LandCoverData as LCD


def train_net(net,
              device,
              tensorboard_writer,
              dir_checkpoint,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              ):
    # 1. Create dataset
    train_set = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/val_index.csv',
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
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    # class weights
    # compute class weights for the loss: inverse-frequency balanced
    # note: we set to 0 the weights for the classes "no_data"(0) and "clouds"(1) to ignore these
    class_weight = (1 / LCD.TRAIN_CLASS_COUNTS[2:]) * LCD.TRAIN_CLASS_COUNTS[2:].sum() / (LCD.N_CLASSES-2)
    class_weight = torch.tensor(np.append(np.zeros(2), class_weight), dtype=torch.float)
    logging.info(f"Will use class weights: {class_weight}, type: {type(class_weight)}")
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # Loss
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                # pbar.set_postfix(**{'loss (batch)': loss.item()})
                writer.add_scalars('Train batch loss', {'loss': loss.item()}, global_step)

                # Evaluation round
                with torch.no_grad():
                    division_step = 60 * batch_size
                    # division_step = 1  # for debugging
                    if division_step > 0:
                        if global_step % division_step == 0:
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                writer.add_histogram('Weights/' + tag, value.data.cpu(), epoch)
                                writer.add_histogram('Gradients/' + tag, value.grad.data.cpu(), epoch)

                            val_score, val_crossentropy_loss, KL_score = evaluate(net, val_loader, device)
                            # for debug
                            # val_score = torch.tensor(0.3081, device='cuda:0')
                            # scheduler.step(val_score)

                            logging.info('Validation Dice score: {}'.format(val_score))
                            writer.add_scalar('Evaluation/learning rate', optimizer.param_groups[0]['lr'], global_step)
                            writer.add_scalar('Evaluation/validation Dice', val_score, global_step)
                            writer.add_scalar('Evaluation/cross entropy loss', val_crossentropy_loss, global_step)
                            writer.add_scalar('Evaluation/validation kl score', KL_score, global_step)

                            # For this training batch
                            i = randrange(batch_size)
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

        writer.add_scalar('Train/Epoch_loss', epoch_loss, epoch)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


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
    # hyper parameters
    args = get_args()
    args.epochs = 50
    args.batch_size = 8
    args.lr = 0.001
    args.tensorboard_dir = '../test2'
    # args.load = '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/test/checkpoint_epoch7.pth'
    # for saving checkpoints
    args.dir_checkpoint = Path('../checkpoints/test2')

    # (Initialize tensor board logging)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels = R-G-B-NIR
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=4, n_classes=10, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  tensorboard_writer=writer,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  dir_checkpoint=args.dir_checkpoint)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
