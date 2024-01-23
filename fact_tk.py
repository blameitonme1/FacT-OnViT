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

def test(model, dl):
    model.eval()
    acc = Accuracy()
    model = model.cuda()
    for batch in dl:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        # 调用avalanche-lib的方法更新
        acc.update(out.argmax(dim=1).view(-1), y, 1)
    
    return acc.result()

def save(args, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in vit.named_parameters():
        if 'FacT' in n or 'head' in n:
            trainable[n] = p.data
    # 保存可训练参数
    torch.save(trainable, 'models/tk/' + args.dataset + '.pt')
    # 保存日志，记录实验过程
    with open('models/tk/' + args.dataset + '.log', 'w') as f:
        f.write(str(ep) + ' ' + str(acc))

def set_seed(seed=0):
    # 手动设置随机性种子，方便结果复现
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config(dataset_name):
    with open('./configs/tk/%s.yaml' % (dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config