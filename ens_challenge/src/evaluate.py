import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils.utils import calculate_kl_loss, write_label_csv

from unet import UNet
from datasets.satellite import SatelliteDataset


@torch.no_grad()
def evaluate(net, dataloader, device):
    """
    Calculate dice score, cross_entropy, KL loss
    """
    net.eval()
    num_val_batches = len(dataloader)
    val_crossentropy_loss_mask = 0
    val_crossentropy_loss_ratio = 0
    mask_crossentropy = torch.nn.CrossEntropyLoss().to(device)
    ratio_crossentropy = torch.nn.CrossEntropyLoss()
    all_image_ids = torch.tensor([])
    true_labels = torch.tensor([])
    predict_labels = torch.tensor([])

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image_ids, image, mask_true, ratio_label = batch['image_id'], batch['image'], batch['mask'], batch['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        # image ids
        all_image_ids = torch.cat((all_image_ids, image_ids))

        # true label
        true_labels = torch.cat((true_labels, ratio_label))

        # predict the mask
        mask_pred = net(image)  # B, C, h, w

        # predicted class ratio by sum of each channel
        batch_ratio = torch.sum(mask_pred, (-2, -1))  # B, C
        batch_ratio /= torch.sum(batch_ratio, 1)[..., None]
        batch_ratio = batch_ratio.cpu()
        predict_labels = torch.cat((predict_labels, batch_ratio))

        # crossentropy_loss on mask
        val_crossentropy_loss_mask += mask_crossentropy(mask_pred, mask_true).float().cpu()
        # crossentropy_loss on ratio
        val_crossentropy_loss_ratio += ratio_crossentropy(batch_ratio, ratio_label)


    write_label_csv(all_image_ids, predict_labels, '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/experiments/evaluation_predict_labels.csv')
    write_label_csv(all_image_ids, true_labels, '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/experiments/evaluation_true_labels.csv')

    # KL_score
    # take into account only last 8 classes
    kl_loss = calculate_kl_loss(predict_labels[:, 2:10], true_labels[:, 2:10])
    net.train()

    return val_crossentropy_loss_mask, val_crossentropy_loss_ratio, kl_loss


if __name__ == '__main__':
    # tensorboard_dir = '../val'
    # writer = SummaryWriter(log_dir=tensorboard_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_set = SatelliteDataset(index_txt_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/val_index.csv',
                               images_folder_path='/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/dataset/',
                               type='val')
    loader_args = dict(batch_size=8, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=4, n_classes=10, bilinear=True).to(device)
    load = '/home/yunfei/Desktop/m2a_sorbonne/ens_challenge/checkpoints/test/checkpoint_epoch6.pth'
    net.load_state_dict(torch.load(load, map_location=device))
    val_crossentropy_loss_mask, val_crossentropy_loss_ratio, kl_loss = evaluate(net, val_loader, device)
    print(f"val_crossentropy_loss_mask: {val_crossentropy_loss_mask}, val_crossentropy_loss_ratio: {val_crossentropy_loss_ratio}, kl_loss: {kl_loss}")
