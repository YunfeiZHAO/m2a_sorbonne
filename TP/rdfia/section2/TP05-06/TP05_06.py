import argparse
import os
import time
from tqdm import tqdm


from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim.lr_scheduler

from tme6 import *

PRINT_INTERVAL = 200


class ConvNet(nn.Module):
    """
    This class defines the structure of the neural network by annonce of tme
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        # We first define the convolution and pooling layers as a features extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0, ceil_mode=True),  # add ceiling to make 7*7 to 4*4
            nn.BatchNorm2d(64)
        )
        # We then define fully connected layers as a classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1000, 10)
            # Reminder: The softmax is included in the loss, do not put it here
        )

    # Method called when we apply the network to an input batch
    def forward(self, input):
        bsize = input.size(0)  # batch size
        output = self.features(input)  # output of the conv layers
        output = output.view(bsize, -1)  # we flatten the 2D feature maps into one 1D vector for each input
        output = self.classifier(output)  # we compute the output of the fc layers
        return output


def get_dataset(batch_size, path, cuda=False):
    """
    This function loads the dataset and performs transformations on each
    image (listed in `transform = ...`).
    """
    train_transforms = transforms.Compose([
        transforms.ToTensor(),  # Transform the PIL image to a torch.Tensor
        transforms.Normalize(mean=(0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201), inplace=True), # mean and std for each channel (total 3 channels)
        transforms.RandomCrop((28,28), padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),  # Transform the PIL image to a torch.Tensor
        transforms.Normalize(mean=(0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201), inplace=True), # mean and std for each channel (total 3 channels)
        transforms.CenterCrop((28,28))
    ])


    train_dataset = datasets.CIFAR10(path, train=True, download=True,
                                     transform=train_transforms)
    val_dataset = datasets.CIFAR10(path, train=False, download=True,
                                   transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False, pin_memory=cuda, num_workers=2)

    return train_loader, val_loader


def epoch(data, model, criterion, optimizer=None, cuda=False):
    """
    Make a pass (called epoch in English) on the data `data` with the
     model `model`. Evaluates `criterion` as loss.
     If `optimizer` is given, perform a training epoch using
     the given optimizer, otherwise, perform an evaluation epoch (no backward)
     of the model.
    """

    # indicates whether the model is in eval or train mode (some layers behave differently in train and eval)
    model.eval() if optimizer is None else model.train()

    # objects to store metric averages
    avg_loss = AverageMeter()
    avg_top1_acc = AverageMeter()
    avg_top5_acc = AverageMeter()
    avg_batch_time = AverageMeter()

    # we iterate on the batches
    tic = time.time()
    for i, (input, target) in enumerate(data):
        if cuda:  # only with GPU, and not with CPU
            input = input.cuda()
            target = target.cuda()

        # forward
        output = model(input)
        loss = criterion(output, target)

        # backward if we are training
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute metrics
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        batch_time = time.time() - tic
        tic = time.time()

        # update
        avg_loss.update(loss.item())
        avg_top1_acc.update(prec1.item())
        avg_top5_acc.update(prec5.item())
        avg_batch_time.update(batch_time)
        if optimizer:
            writer.add_scalar('Train batch loss', avg_loss.val, i+1)
        # print info
        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:5.1f} ({top1.avg:5.1f})\t'
                  'Prec@5 {top5.val:5.1f} ({top5.avg:5.1f})'.format(
                "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                top1=avg_top1_acc, top5=avg_top5_acc))
    writer.add_graph(model, input)

    # Print summary
    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg Prec@1 {top1.avg:5.2f} %\t'
          'Avg Prec@5 {top5.avg:5.2f} %\n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        top1=avg_top1_acc, top5=avg_top5_acc))

    return avg_top1_acc, avg_top5_acc, avg_loss


def main(params):
    # define model, loss, optim
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), params.lr, params.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    global lr_sched
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.scheduler_gamma)

    if params.cuda:  # si on fait du GPU, passage en CUDA
        cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()
    # On récupère les données
    train, test = get_dataset(params.batch_size, params.path, params.cuda)

    # On itère sur les epochs
    for i in range(params.epochs):
        print("=================\n=== EPOCH " + str(i + 1) + " =====\n=================\n")
        # Phase de train
        top1_acc, avg_top5_acc, loss = epoch(train, model, criterion, optimizer, cuda=params.cuda)
        # Phase d'evaluation
        top1_acc_test, top5_acc_test, loss_test = epoch(test, model, criterion, cuda=params.cuda)
        # change learning rate
        writer.add_scalar('lr', lr_sched.get_last_lr()[0], i + 1)
        lr_sched.step()
        # tensorboard plot
        writer.add_scalars('Loss/Epoch train and test LOSS',
                           {'Train loss': loss.avg,
                            'Test loss': loss_test.avg},
                           i + 1)

        writer.add_scalars('Accuracy/Epoch train and test TOP1 ACCURACY',
                           {'Train top1 accuracy': top1_acc.avg,
                            'Test top1 accuracy': top1_acc_test.avg},
                           i + 1)


if __name__ == '__main__':
    # Parameters
    args = load_yaml('configs/cifar-10.yaml')

    # tensorboard
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
    outdir = "./experiments" + "/" + args.name + "_" + date_time
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    global writer
    writer = SummaryWriter(outdir)

    # run main
    main(args)
    print("done")
