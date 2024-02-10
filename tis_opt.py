import torch
from torch import nn
from torch.utils.data import DataLoader
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
from slice import *
import loralib as lora

def train(args, model, dl, opt, scheduler, epoch, printDim=False):
    # args.best_acc = 0
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            if printDim:
                print(out.shape, y.shape)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(vit, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
                save(args, model, acc, ep)
            pbar.set_description(str(acc) + '|' + str(args.best_acc))

    model = model.cpu()
    return model, acc

@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    # pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)

    return acc.result()


def lora_forward_attn_full(self, x):
    """ LoRA设置q, k, v, o的时候的forward函数 """
    B, N, C = x.shape
    qkv = self.qkv(x)
    q = self.q_LoRA(x)
    k = self.k_LoRA(x)
    v = self.v_LoRA(x)
    qkv += torch.cat([q, k, v], dim=2)
    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj += self.proj_LoRA(x)
    x = self.proj_drop(proj)
    return x

def lora_forward_attn_qv(self, x):
    """ LoRA设置q, k, v, o的时候的forward函数 """
    B, N, C = x.shape
    qkv = self.qkv(x)
    q = vit.q_LoRA(x)
    k = torch.zeros([B, N, C]) # 增量是一个零矩阵，这样就相当于不更新W_k了
    v = vit.v_LoRA(x)
    qkv += torch.cat([q, k, v], dim=2)
    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    # proj += vit.proj_LoRA(x) 也不再更新W_o了
    x = self.proj_drop(proj)
    return x

def lora_forward_mlp(self, x):
    """ 之后可能会用到，先留在这边 """
    B, N, C = x.shape
    h = self.fc1(x)  # B n 4c
    h += self.fc1_LoRA(x)
    x = self.act(h)
    x = self.drop(x)
    h = self.fc2(x)
    h += self.fc2_LoRA(x)
    x = self.drop(h)
    return x

def fact_forward_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)

    q = vit.FacTv(self.dp(self.q_FacTs(vit.FacTu(x))))
    k = vit.FacTv(self.dp(self.k_FacTs(vit.FacTu(x))))
    v = vit.FacTv(self.dp(self.v_FacTs(vit.FacTu(x))))

    qkv += torch.cat([q, k, v], dim=2) * self.s

    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj += vit.FacTv(self.dp(self.proj_FacTs(vit.FacTu(x)))) * self.s
    x = self.proj_drop(proj)
    return x


def fact_forward_mlp(self, x):
    B, N, C = x.shape
    h = self.fc1(x)  # B n 4c


    h += vit.FacTv(self.dp(self.fc1_FacTs(vit.FacTu(x))).reshape(
        B, N, 4, self.dim)).reshape(
        B, N, 4 * C) * self.s
    x = self.act(h)
    x = self.drop(x)
    h = self.fc2(x)
    x = x.reshape(B, N, 4, C)
    h += vit.FacTv(self.dp(self.fc2_FacTs(vit.FacTu(x).reshape(
        B, N, 4 * self.dim)))) * self.s
    x = self.drop(h)
    return x


def set_FacT(model, dim=8, s=1):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.FacTu = nn.Linear(768, dim, bias=False)
        model.FacTv = nn.Linear(dim, 768, bias=False)
        model.dim = dim # 储存当前的rank
        nn.init.zeros_(model.FacTv.weight)
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            _.q_FacTs = nn.Linear(dim, dim, bias=False)
            _.k_FacTs = nn.Linear(dim, dim, bias=False)
            _.v_FacTs = nn.Linear(dim, dim, bias=False)
            _.proj_FacTs = nn.Linear(dim, dim, bias=False)
            _.dp = nn.Dropout(0.1)
            _.s = s
            _.dim = dim
            bound_method = fact_forward_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif type(_) == timm.models.layers.mlp.Mlp:
            _.fc1_FacTs = nn.Linear(dim, dim * 4, bias=False)
            _.fc2_FacTs = nn.Linear(4 * dim, dim, bias=False)
            _.dim = dim
            _.s = s
            _.dp = nn.Dropout(0.1)
            bound_method = fact_forward_mlp.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_FacT(_, dim, s)


def set_LoRA_full(model, dim=8, s=1):
    """ 对所有的自注意力权重进行LoRA """
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        pass # 此时不需要做任何事情
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            _.q_LoRA = lora.Linear(model_dim, model_dim, r=dim)
            _.k_LoRA = lora.Linear(model_dim, model_dim, r=dim)
            _.v_LoRA = lora.Linear(model_dim, model_dim, r=dim)
            _.proj_LoRA = lora.Linear(model_dim, model_dim, r=dim)
            bound_method = lora_forward_attn_full.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif type(_) == timm.models.layers.mlp.Mlp:
            """ 只考虑自注意力的权重 """
            # _.fc1_LoRA = lora.Linear(model_dim, model_dim * 4, r=dim)
            # _.fc2_LoRA = lora.Linear(4 * model_dim, model_dim, r=dim)
            # bound_method = lora_forward_mlp.__get__(_, _.__class__)
            # setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_LoRA_full(_, dim, s)


def set_LoRA_qv(model, dim=8, s=1):
    """ 对qv进行LoRA """
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        pass # 此时不需要做任何事情
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            _.q_LoRA = lora.Linear(model_dim, model_dim, r=dim)
            _.v_LoRA = lora.Linear(model_dim, model_dim, r=dim)
            bound_method = lora_forward_attn_qv.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif type(_) == timm.models.layers.mlp.Mlp:
            """ 只考虑自注意力的权重"""
            # _.fc1_LoRA = lora.Linear(model_dim, model_dim * 4, r=dim)
            # _.fc2_LoRA = lora.Linear(4 * model_dim, model_dim, r=dim)
            # bound_method = lora_forward_mlp.__get__(_, _.__class__)
            # setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_LoRA_qv(_, dim, s)

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config(dataset_name):
    with open('./configs/tt/%s.yaml' % (dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


@torch.no_grad()
def save(args, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in vit.named_parameters():
        if 'FacT' in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, 'models/tt/' + args.dataset + '.pt')
    with open('models/tt/' + args.dataset + '.log', 'w') as f:
        f.write(str(ep) + ' ' + str(acc))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dim', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    args = parser.parse_args()
    print(args)
    seed = args.seed
    set_seed(seed)
    name = args.dataset
    args.best_acc = 0
    vit = create_model(args.model, checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    model_dim = vit.embed_dim
    train_dl, test_dl = get_data(name)
    config = get_config(name)
    if args.dim == 0:
        args.dim = config['rank']
    if args.scale == 0:
        args.scale = config['scale']
    # set_FacT(vit, dim=args.dim, s=args.scale)
    set_LoRA_full(vit, dim=args.dim, s=args.scale) # 两个方法选择一个
    trainable = []
    vit.reset_classifier(get_classes_num(name))
    total_param = 0
    for n, p in vit.named_parameters():
        if ('lora' in n or 'head' in n) and (p.requires_grad is True):
            trainable.append(p)
            if 'head' not in n and (p.requires_grad is True):
                # print(f" name {n}, num {p.numel()}")
                # print(f"name {n} shape {p.shape}")
                total_param += p.numel()
        else:
            p.requires_grad = False
    print(f"total_param is {total_param}")
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
    # if opt == None:
    #     print("error")
    # vit = rank_descend(args, vit, train_dl)
    vit, cur_acc = train(args, vit, train_dl, opt, scheduler, epoch=40)
    print(f"cur acc is {cur_acc}")
    best_acc = args.best_acc # 记录出现的最高精度

    while True:
        vit = freeze_LoRA(vit)
        # 重新统计此时参数数量
        trainable = []
        total_param = 0
        for n, p in vit.named_parameters():
            if ('LoRA' in n) and (p.requires_grad is True):
                trainable.append(p)
                if 'head' not in n and (p.requires_grad is True):
                    # print(f"1 name {n}, num {p.numel()}")
                    total_param += p.numel()
            else:
                p.requires_grad = False
        print(f"total_param is {total_param}")
        # opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        # scheduler = CosineLRScheduler(opt, t_initial=100,
        #                           warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
        vit, cur_acc = train(args, vit, train_dl, opt, scheduler, epoch=10) # 先训练10个epoch先
        for i in range(10):
            vit, cur_acc = train(args, vit, train_dl, opt, scheduler, epoch=10)
        # past_acc = cur_acc
        # while past_acc <= cur_acc:
        #     # acc保持上升的时候
        #     vit, cur_acc = train(args, vit, train_dl, opt, scheduler, epoch=30)
        #     if past_acc > cur_acc:
        #         break
        #     else:
        #         past_acc = cur_acc
        # # 此时past_acc就是没有过拟合时候最高的精度
        # print(f"bset acc during this epoch is {past_acc}")
        # # 查看最高acc是否增加，不增加退出循环
        # if past_acc >= best_acc:
        #     best_acc = past_acc
        # else:
        #     break
    # showDim(vit)
    # print(f"trace back acc is {test(vit, test_dl)[1]}")
    vit = train(args, vit, train_dl, opt, scheduler, epoch=10)[0]
    print('acc1:', args.best_acc)
    print(f"optimal rank is {vit.dim}")