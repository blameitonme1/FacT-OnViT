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

def calculate_taylor(parameter):
    if not parameter.requires_grad:
        print("error when grad is None")
        return 0
    if parameter.grad is not None:
        taylor = parameter * parameter.grad
        return taylor.mean()
    else:
        print("error when grad is None")
        return 0

def calculate_freeze_candidate_FacT(model, num):
    taylors = {}
    freeze_candidate = None
    for n, p in model.named_parameters():
        # 选择范围是从trainable里面
        if 'FacT' in n and p.requires_grad is True and 'FacTu' not in n:
            taylors[n] = calculate_taylor(p)
    # 按照值进行排序
    sorted_dict_by_value = dict(sorted(taylors.items(), key=lambda item: item[1]))
    freeze_candidate = list(sorted_dict_by_value.keys())[:num]
    return freeze_candidate

def calculate_freeze_candidate_LoRA(model, num):
    taylors = {}
    freeze_candidate = None
    for n, p in model.named_parameters():
        # 选择范围是从trainable里面
        if ('LoRA' in n) and p.requires_grad is True:
            taylors[n] = calculate_taylor(p)
    # 按照值进行排序
    sorted_dict_by_value = dict(sorted(taylors.items(), key=lambda item: item[1]))
    freeze_candidate = list(sorted_dict_by_value.keys())[:num]
    # print(sorted_dict_by_value)
    return freeze_candidate

def randomly_freeze_FacT(model, num):
    # 获取模型中包含 'FacT' 的参数列表
    fact_parameters = {n: p for n, p in model.named_parameters() if 'FacT' in n and p.requires_grad}
    # 确保选择的参数个数不超过实际存在的 'FacT' 参数个数
    num = min(num, len(fact_parameters))
    # 从 'FacT' 参数列表中随机选择 num 个参数
    freeze_candidate_names = random.sample(fact_parameters.keys(), num)
    # 冻结选定的参数
    for name in freeze_candidate_names:
        param = fact_parameters[name]
        param.requires_grad = False
    return freeze_candidate_names


def freeze_FacT(model):
    freeze_candidate = calculate_freeze_candidate_FacT(model, num=4)
    print(len(freeze_candidate))
    for n, p in model.named_parameters():
        if n in freeze_candidate:
            # 冻结该参数
            p.requires_grad = False
    return model

def freeze_LoRA(model):
    freeze_candidate = calculate_freeze_candidate_LoRA(model, num=4)
    print(len(freeze_candidate))
    for n, p in model.named_parameters():
        if n in freeze_candidate:
            # 冻结该参数
            print(n)
            p.requires_grad = False
    return model

def showDim_FacT(model):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        print(f"U shape {model.FacTu.weight.shape}")
        print(f"V shape {model.FacTv.weight.shape}")
        print(f"dim is {model.dim}")
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            print(f"q shape {_.q_FacTs.weight.shape}")
            print(f"k shape {_.k_FacTs.weight.shape}")
            print(f"v shape {_.v_FacTs.weight.shape}")
            print(f"proj shape {_.proj_FacTs.weight.shape}")
            print(f"attention dim is {_.dim}")
        elif type(_) == timm.models.layers.mlp.Mlp:
            print(f"fc1 shape {_.fc1_FacTs.weight.shape}")
            print(f"fc2 shape {_.fc2_FacTs.weight.shape}")
            print(f"ffn dim is {_.dim}")
        elif len(list(_.children())) != 0:
            showDim_FacT(_)