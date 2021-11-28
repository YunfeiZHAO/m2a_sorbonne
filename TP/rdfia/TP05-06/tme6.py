import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import yaml

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # maxk, batch_size
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # copy target to much pred, get boolean matrix maxk, batch_size

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size)) # add all true and divide by batch size
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainLossPlot(object):
    def __init__(self):
        self.loss_train = []
        self.fig = plt.figure()

    def update(self, loss_train):
        self.loss_train.append(loss_train)

    def plot(self):
        plt.figure(self.fig.number)
        plt.clf()
        plt.plot(np.array(self.loss_train))
        plt.title("Train loss / batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.show()
        plt.draw_all()
        plt.pause(1e-3)

class AccLossPlot(object):
    def __init__(self):
        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []
        self.fig = plt.figure()

    def update(self, loss_train, loss_test, acc_train, acc_test):
        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        self.acc_train.append(acc_train)
        self.acc_test.append(acc_test)
        plt.figure(self.fig.number)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(np.array(self.acc_train), label="acc. train")
        plt.plot(np.array(self.acc_test), label="acc. test")
        plt.title("Accuracy / epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.array(self.loss_train), label="loss train")
        plt.plot(np.array(self.loss_test), label="loss test")
        plt.title("Loss / epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.draw_all()
        plt.pause(1e-3)


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml(path):
    with open(path, 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.Loader)
    return DotDict(opt)


def write_yaml(file, dotdict):
    d = dict(dotdict)
    with open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(d, outfile, default_flow_style=False, allow_unicode=True)