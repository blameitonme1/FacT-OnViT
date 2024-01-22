import torch
from torch import nn
from torch.utils.data import Dataloader
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
import numpy as np
import random
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
import yaml

def train(args, model, dl, opt, scheduler, epoch):
    # dl -> dataloader
    # opt -> optimizer
    # scheduler用来自动调节学习率
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            # x是图像向量， y是标签向量
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if scheduler is not None:
            # 有设置调节学习率
            scheduler.step(ep)
        
        if ep % 10 == 9:
            acc = test(vit, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
                save(args, model, acc, ep)
            pbar.set_description(str(acc) + '|' + str(args.best_acc))
    
    # 回到CPU方便用numpy对参数操作
    model = model.cpu()
    return model

def test():
    pass
            