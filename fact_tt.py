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
            print(acc)
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
    torch.save(trainable, 'models/tt/' + args.dataset + '.pt')
    # 保存日志，记录实验过程
    with open('models/tt/' + args.dataset + '.log', 'w') as f:
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

def fact_forward_attn(self, x):
    # 在FacT框架下的自注意力多头池化
    B, N, C = x.shape
    qkv = self.qkv(x) # pretrained模型的qkv
    # 计算查询（q）、键（k）和值（v）的投影, 暂时当作抽象黑盒
    q = vit.FacTv(self.dp(self.q_FacTs(vit.FacTu(x))))
    k = vit.FacTv(self.dp(self.k_FacTs(vit.FacTu(x))))
    v = vit.FacTv(self.dp(self.v_FacTs(vit.FacTu(x))))
    # 将查询、键、值投影与原始的qkv进行拼接，并乘以缩放因子s
    qkv += torch.cat([q, k, v], dim=2) * self.s
    # 3 代表 q，k，v，而num_heads处理多头注意力
    qkv = qkv.reshape(
        B, N, 3, self.num_heads, C // self.num_heads
    ).permute(2, 0, 3, 1, 4) # 多头注意避免使用for循环，不懂看FacT笔记
    q, k, v = qkv[0], qkv[1], qkv[2]
    # 缩放点积注意力， 注意-2和-1，因为前面还有别的维度
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # attn, v的shape：（B, num_heads, N, C // self.num_heads）
    # 这样X就回到B, N, C,其中C是经过全连接投影之后的元素长度，没有改变，因为是(C, C)的全连接
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x) # 原本的被冻结的权重W_o的forward结果
    # 实现残差连接,dp是dropout
    proj += vit.FacTv(self.dp(self.proj_FacTs(vit.FacTu(x)))) * self.s
    x = self.proj_drop(proj)
    return x

def fact_forward_mlp(self, x):
    # FacT框架下进行FFN,比平常FFN多了残差和dropout，Factu和FacTv分别是论文decompose之后的U和V向量
    B, N, C = x.shape
    h = self.fc1(x)
    h += vit.FacTv(self.dp(self.fc1_FacTs(vit.FacTu(x))).reshape(B, N, 4, self.dim)).reshape(
        B, N, 4 * C
    ) * self.s
    x = self.act(h) # 激活函数
    x = self.drop(x)
    h = self.fc2(x)
    x = x.reshape(B, N, 4, C)
    h += vit.FacTv(self.dp(self.fc2_FacTs(vit.FacTu(x).reshape(
        B, N, 4 * self.dim
    )))) * self.s
    x = self.drop(h)
    return x

def set_FacT(model, dim=8, s=1):
    # s是缩放因子， dim是论文的r
    # 768是16x16x3，因为一个patch是16x16，并且有三个通道（RGB）
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.FacTu = nn.Linear(768, dim, bias=False)
        model.FacTv = nn.Linear(dim, 768, bias=False)
        nn.init.zeros_(model.FacTv.weight) # 论文V初始化为0
    for _ in model.children():
        # 遍历模型各个模块
        if type(_) == timm.models.vision_transformer.Attention:
            # 注意力模块,这四个层对应了论文的SIGMAX
            _.q_FacTs = nn.Linear(dim, dim, bias=False)
            _.k_FacTs = nn.Linear(dim, dim, bias=False)
            _.v_FacTs = nn.Linear(dim, dim, bias=False)
            _.proj_FacTs = nn.Linear(dim, dim, bias=False)
            _.dp = nn.Dropout(0.1)
            _.s = s
            _.dim = dim
            bound_method = fact_forward_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method) # 修改forward的时候应该进行的函数
        
        elif type(_) == timm.models.layers.mlp.Mlp:
            # 这里注意一下
            _.fc1_FacTs = nn.Linear(dim, dim * 4, bias=False)
            _.fc2_FacTs = nn.Linear(4 * dim, dim, bias=False)
            _.dim = dim
            _.s = s
            _.dp = nn.Dropout(0.1)
            bound_method = fact_forward_mlp.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            # 递归遍历处理transformer块的子模块
            set_FacT(_, dim, s)
        
def get_config(dataset_name):
    with open('./configs/tt/%s.yaml' % (dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    parser = ArgumentParser()
    # 加入命令行参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dim', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4) # weight decay
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    args = parser.parse_args() # 解析并存放参数
    print(args)
    seed = args.seed
    set_seed(seed)
    name = args.dataset
    args.best_acc = 0
    vit = create_model(args.model, checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(name) # 得到dataloader
    config = get_config(name)
    # 如果没有特地指明，使用配置文件的配置
    if args.dim == 0:
        args.dim = config['rank']
    if args.scale == 0:
        args.scale = config['scale']
    set_FacT(vit, dim=args.dim, s=args.scale)

    trainable = []
    vit.reset_classifier(get_classes_num(name)) # 设置模型分类器分类个数
    total_param = 0
    for n, p in vit.named_parameters():
        # 我推测head是分类器的参数名字
        if 'FacT' in n or 'head' in n:
            trainable.append(p)
            if 'head' not in n:
                total_param += p.numel()
        else:
            # 不是选择的参数，直接冻结,（就是冻结transformer的pretrained的模型）
            p.requires_grad = False
    
    print('total_param',total_param)
    # 只把trainable传递给opt，会计算梯度的只有这些参数（decompose之后的矩阵）
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    vit = train(args, vit, train_dl, opt, scheduler, epoch=100)
    print('acc1:', args.best_acc)







    

    
            