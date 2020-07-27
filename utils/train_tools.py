import torch
import torch.nn as nn
import os
import glob


def initilize_model(model):     # <--Modified by Mingyang Wang 20180814: Before training, you should initialize all the parameters in your constucted network
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_normal(model.weight.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def delete_existed_params(dir):
    existed_params = os.listdir(dir)
    if len(existed_params) > 10:
        min_epoch = min([eval(item.split(':_')[1].split('_')[0]) for item in existed_params])
        path = glob.glob('{}/*_{}_*.pkl'.format(dir, min_epoch))[0]
        os.remove(path)


