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

def backward_hook(module, grad_input, grad_output):
    # print(f"Gradient for module {module.__class__.__name__}: ", grad_input)
    total_params = sum(p.numel() for p in module.parameters())

def rank_descend(args, model, dl):
    global opt
    global scheduler
    model = train(args, model, dl, opt, scheduler, epoch=10)[0]
    best_acc = args.best_acc
    cur_acc = best_acc
    while True:
        print(f"cur_acc is {cur_acc}, best_acc = {best_acc}")
        if cur_acc < best_acc:
            # 减少了rank训练还是不能回复正确率，说明rank已经到了最佳值了
            # 加载保存的模型参数
            loaded_trainable = torch.load('models/tt/' + args.dataset + '.pt')
            # 将加载的参数设置回模型中
            for n, p in model.named_parameters():
                if n in loaded_trainable:
                    p.data = loaded_trainable[n]
            break
        freeze_FacT(model)
        show_trinable_and_updateOpt(model)
        model = model.cuda()
        model, cur_acc = train(args, model, dl, opt, scheduler, epoch=10)
        save(args, model, cur_acc)
        if cur_acc > best_acc:
            best_acc = cur_acc
    return model

def show_trinable_and_updateOpt(model):
    global opt
    global scheduler
    trainable = []
    total_param = 0
    for n, p in vit.named_parameters():
        if 'FacT' in n or 'head' in n:
            trainable.append(p)
            if 'head' not in n:
                # print(f"1 name {n}, num {p.numel()}")
                total_param += p.numel()
        else:
            p.requires_grad = False
    print(f"total_param is {total_param}, rank is {model.dim} now")
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)

def train(args, model, dl, opt, scheduler, epoch):
    args.best_acc = 0
    model = model.cuda()
    model.train()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
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
            pbar.set_description(str(acc) + '|' + str(args.best_acc))

    model = model.cpu()
    return model, args.best_acc

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
        # model.FacTu.register_backward_hook(backward_hook)
        # model.FacTv.register_backward_hook(backward_hook)
        model.dim = dim # 储存当前的rank
        nn.init.zeros_(model.FacTv.weight)
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            _.q_FacTs = nn.Linear(dim, dim, bias=False)
            _.k_FacTs = nn.Linear(dim, dim, bias=False)
            _.v_FacTs = nn.Linear(dim, dim, bias=False)
            _.proj_FacTs = nn.Linear(dim, dim, bias=False)
            # _.q_FacTs.register_backward_hook(backward_hook)
            # _.k_FacTs.register_backward_hook(backward_hook)
            # _.v_FacTs.register_backward_hook(backward_hook)
            # _.proj_FacTs.register_backward_hook(backward_hook)
            _.dp = nn.Dropout(0.1)
            _.s = s
            _.dim = dim
            bound_method = fact_forward_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif type(_) == timm.models.layers.mlp.Mlp:
            _.fc1_FacTs = nn.Linear(dim, dim * 4, bias=False)
            _.fc2_FacTs = nn.Linear(4 * dim, dim, bias=False)
            # _.fc1_FacTs.register_backward_hook(backward_hook)
            # _.fc2_FacTs.register_backward_hook(backward_hook)
            _.dim = dim
            _.s = s
            _.dp = nn.Dropout(0.1)
            bound_method = fact_forward_mlp.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_FacT(_, dim, s)


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
        f.write(str(ep) + ' ' + str(acc) + ' ' + str(model.dim))


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
    train_dl, test_dl = get_data(name)
    config = get_config(name)
    if args.dim == 0:
        args.dim = config['rank']
    if args.scale == 0:
        args.scale = config['scale']
    set_FacT(vit, dim=args.dim, s=args.scale)

    
    opt = None
    scheduler = None
    show_trinable_and_updateOpt(vit)
    if opt == None:
        print("error")
    vit = rank_descend(args, vit, train_dl)
    vit = train(args, vit, train_dl, opt, scheduler, epoch=20)[0]
    print('acc1:', args.best_acc)
    print(f"optimal rank is {vit.dim}")