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
def prune_U_byRank(module):
    outDim, inDim = module.weight.shape
    # 获取梯度的绝对值
    weight = module.weight.transpose(0, 1)
    gradient_copy = module.weight.grad.clone().abs()
    gradient_copy = gradient_copy.transpose(0, 1)
    taylor = gradient_copy * weight # 计算1阶泰勒展开, 忽略剩余项
    # 逐行处理
    sliced_weights = []
    for i in range(inDim):
        # 获取当前行的梯度
        current_row_grad = taylor[i, :]
        # 找到值最小的元素的索引
        min_index = torch.argmin(current_row_grad)
        # 删除值最小的元素
        if min_index < outDim:
            sliced_row = torch.cat((weight[i, :min_index], weight[i, min_index+1:]))
        else:
            sliced_row = weight[i, :min_index]
        # 将处理后的行添加到列表中
        sliced_weights.append(sliced_row)
    # 将列表转换为张量
    sliced_weight = torch.stack(sliced_weights)
    sliced_weight = sliced_weight.transpose(0, 1)
    return sliced_weight

def prune_V_byRank(module):
    inDim, outDim = module.weight.shape
    # 复制梯度，确保不修改原始梯度张量
    gradient_copy = module.weight.grad.clone().abs()
    taylor = gradient_copy * module.weight # 计算1阶泰勒展开
    # 逐行处理
    sliced_weights = []
    for i in range(inDim):
        # 获取当前行的梯度
        current_row_grad = taylor[i, :]
        # 找到值最小的元素的索引
        min_index = torch.argmin(current_row_grad)
        # 删除值最小的元素
        if min_index < outDim:
            sliced_row = torch.cat((module.weight[i, :min_index], module.weight[i, min_index+1:]))
        else:
            sliced_row = module.weight[i, :min_index]
        # 将处理后的行添加到列表中
        sliced_weights.append(sliced_row)
    # 将列表转换为张量
    sliced_weight = torch.stack(sliced_weights)
    return sliced_weight


def prune_qkv_byRank(module):
    dim = module.weight.shape[0]
    newDim = dim - 1
    sliced_weight = []
    for i in range(newDim):
        sliced_row = module.weight[i, :newDim]
        sliced_weight.append(sliced_row)
    
    # 将列表转换为张量
    sliced_weight_tensor = torch.stack(sliced_weight)
    return sliced_weight_tensor

def prune_fc_byRank(weight):
    inDim, outDim = weight.shape[0], weight.shape[1]
    newInDim = inDim - 1
    newOutDim = outDim - 4
    r = outDim / 4
    r = int(r)
    sliced_weight = []
    for i in range(newInDim):
        sliced_row = weight[i, :r - 1]
        sliced_row = torch.cat([sliced_row, weight[i, r:2 * r - 1]])
        sliced_row = torch.cat([sliced_row, weight[i, 2 * r:3 * r - 1]])
        sliced_row = torch.cat([sliced_row, weight[i, 3 * r:4 * r - 1]])
        sliced_weight.append(sliced_row)
    sliced_weight_tensor = torch.stack(sliced_weight)
    return sliced_weight_tensor

def initialize_module_with_sliced_weight(module, sliced_weight):
    # 将新模型的权重设置为 sliced_weight
    with torch.no_grad():
        module.weight = torch.nn.Parameter(sliced_weight)
    module.requires_grad = True
    # 将新模型的状态字典加载到原始模型中
    module.load_state_dict(module.state_dict())




def trace_back_model(args, model):
    loaded_trainable = torch.load('models/tt/' + args.dataset + '_bestrank.pt')
            # 将加载的参数设置回模型中
    for n, p in model.named_parameters():
        if n in loaded_trainable:
            print(f"{n}:")
            print(p.shape)
            p.data = loaded_trainable[n]
    

def trace_back_FacT_setting(model):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.dim = model.dim + 1 # rank减少了
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            _.dim += 1
        elif type(_) == timm.models.layers.mlp.Mlp:
            _.dim += 1
        elif len(list(_.children())) != 0:
            trace_back_FacT_setting(_)
    
def prune_FacT(model):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        U_sliced = prune_U_byRank(model.FacTu)
        model.FacTu = nn.Linear(U_sliced.shape[1], U_sliced.shape[0], bias=False)
        initialize_module_with_sliced_weight(model.FacTu, U_sliced)
        V_sliced = prune_V_byRank(model.FacTv)
        model.Factv = nn.Linear(V_sliced.shape[0], V_sliced.shape[1], bias=False)
        initialize_module_with_sliced_weight(model.FacTv, V_sliced)
        model.dim = model.dim - 1 # rank减少了
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            q_sliced = prune_qkv_byRank(_.q_FacTs)
            _.q_FacTs = nn.Linear(q_sliced.shape[0], q_sliced.shape[1], bias=False)
            initialize_module_with_sliced_weight(_.q_FacTs, q_sliced)
            k_sliced = prune_qkv_byRank(_.k_FacTs)
            _.k_FacTs = nn.Linear(k_sliced.shape[0], k_sliced.shape[1], bias=False)
            initialize_module_with_sliced_weight(_.k_FacTs, k_sliced)
            v_sliced = prune_qkv_byRank(_.v_FacTs)
            _.v_FacTs = nn.Linear(v_sliced.shape[0], v_sliced.shape[1], bias=False)
            initialize_module_with_sliced_weight(_.v_FacTs, v_sliced)
            proj_sliced = prune_qkv_byRank(_.proj_FacTs)
            _.proj_FacTs = nn.Linear(proj_sliced.shape[0], proj_sliced.shape[1], bias=False)
            initialize_module_with_sliced_weight(_.proj_FacTs, proj_sliced)
            _.dim -= 1
        elif type(_) == timm.models.layers.mlp.Mlp:
            # 这里转置一下
            fc1_sliced = prune_fc_byRank(_.fc1_FacTs.weight.transpose(0, 1))
            # print(f"fc1 {fc1_sliced.shape}")
            _.fc1_FacTs = nn.Linear(fc1_sliced.shape[0], fc1_sliced.shape[1], bias=False)
            initialize_module_with_sliced_weight(_.fc1_FacTs, fc1_sliced.transpose(0, 1))
            fc2_sliced = prune_fc_byRank(_.fc2_FacTs.weight)
            # print(f"fc2 {fc1_sliced.shape}")
            _.fc2_FacTs = nn.Linear(fc2_sliced.shape[1], fc2_sliced.shape[0], bias=False)
            initialize_module_with_sliced_weight(_.fc2_FacTs, fc2_sliced)
            _.dim -= 1
        elif len(list(_.children())) != 0:
            prune_FacT(_)

def showDim(model):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        print(f"U shape {model.FacTu.weight.shape}")
        print(f"V shape {model.FacTv.weight.shape}")
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            print(f"q shape {_.q_FacTs.weight.shape}")
            print(f"k shape {_.k_FacTs.weight.shape}")
            print(f"v shape {_.v_FacTs.weight.shape}")
            print(f"proj shape {_.proj_FacTs.weight.shape}")
        elif type(_) == timm.models.layers.mlp.Mlp:
            print(f"fc1 shape {_.fc1_FacTs.weight.shape}")
            print(f"fc2 shape {_.fc2_FacTs.weight.shape}")
        elif len(list(_.children())) != 0:
            showDim(_)