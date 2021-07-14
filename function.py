import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import upsample, interpolate
from Spa_downs import *
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from torch.nn import init
import scipy.io as sio
import os
import random


class ReshapeTo2D(nn.Module):

    def __init__(self):
        super(ReshapeTo2D, self).__init__()

    def forward(self,x):
        return torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))

class ReshapeTo3D(nn.Module):
    def __init__(self):
        super(ReshapeTo3D, self).__init__()

    def forward(self,x):
        return  torch.reshape(x, (x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))))

class TransDimen(nn.Module):
    def __init__(self):
        super(TransDimen, self).__init__()

    def forward(self,x):
        return torch.Tensor.permute(x,[0,2,1])

def channel_crop(data, position, length):
    assert data.size(1) >= position + length, 'the cropped channel out of size.'
    return data[:, position: position + length, :, :]

def ins (list_, data, index):
    list_start = list_[:index]
    list_start = [ Variable(i, requires_grad=False).type(torch.cuda.FloatTensor) for i in list_start]
    data = [Variable(data, requires_grad=False).type(torch.cuda.FloatTensor)]
    list_end = list_[index:]
    list_end = [ Variable(i, requires_grad=False).type(torch.cuda.FloatTensor) for i in list_end]

    return list_start + data + list_end

def to_gpu(data):
    return Variable(data, requires_grad=False).type(torch.cuda.FloatTensor)


class L_Dspec(nn.Module):
    def __init__(self,in_channel,out_channel,P_init):
        super(L_Dspec, self).__init__()
        self.in_channle = in_channel
        self.out_channel = out_channel
        self.P = nn.Parameter(P_init)

    def forward(self,input):
        S = input.shape
        out = torch.reshape(input,[S[0],S[1],S[2]*S[3]])
        out = torch.matmul(self.P,out)

        return torch.reshape(out,[S[0],self.out_channel,S[2],S[3]])

def add_wgn(x, snr):
    P_signal=torch.sum(x.abs()**2)
    P_noise = P_signal/10**(snr/10.0)
    sigma = torch.sqrt(P_noise/x.numel())
    noise = torch.randn(x.shape).type(torch.cuda.FloatTensor)
    return x + sigma * noise

def tensor_copy(x):
    return x.clone()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model'     , default='MSDANet',        help='MSDANet')
    parser.add_argument('--fusion'    , default='Concate',        help='Concate')
    parser.add_argument('--lr'        , default=1e-4, type=float, help='learning rate for optimizer')
    parser.add_argument('--batch_size', default=16, type=int,     help='batch size for training')
    parser.add_argument('--factor'    , default=8, type=int,      help='scale factor. 4/8/16')
    parser.add_argument('--dataset'   , default='Houston',        help='Houston/PaviaU/dc/PaviaC')
    parser.add_argument('--patch_size', default=64, type=int,     help='patch size of training')
    parser.add_argument('--stride'    , default=32, type=int,     help='stride of training')
    parser.add_argument('--pan'       , action='store_true',      help='pan_sharpening or MSI + HSI')
    parser.add_argument('--mem_load'  , action='store_true',      help='load the all dataset into memory or disk')
    parser.add_argument('--phase'     , default='train',          help='train/test')
    parser.add_argument('--noise'     , action='store_true',      help='wheater to add noise to LR_HSI and HR_MSI')

    return parser.parse_args()


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)

def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    random.seed(4)
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2


def all_data_in(Path='Data/Houston/', datasets='Houston', Train_image_num=10):

    names = get_img_name(Path=Path, datasets=datasets)
    allData = []

    for i in range(Train_image_num):
        Data = sio.loadmat(os.path.join(Path, names[i])+'.mat')
        HSI = Data['hsi']
        HSI = HSI.transpose((2, 0, 1))
        allData.append(HSI)

    return allData

dataset_dict = dict(
    PaviaC = [10, 5, 300, 8000, 102, 1, (55, 41, 12)], ### [train_img_num, val_img_num, stop epoch, max_value, band_number, RGB]
    PaviaU = [10, 5, 300, 8000, 103, 1, (46, 27, 10)],
    Houston = [3, 2, 300, 65535, 144, 1, (65, 51, 22)],
    dc = [11, 5, 300, 65535, 191, 4, (51, 35, 21)],
    )


def get_img_name(Path='Data/Houston/', datasets='Houston'):

    names_PaviaC_list = [
    'PaviaC_11', 'PaviaC_12', 'PaviaC_13', 'PaviaC_21', 'PaviaC_22', 'PaviaC_23',
    'PaviaC_31', 'PaviaC_32', 'PaviaC_33', 'PaviaC_41', 'PaviaC_42', 'PaviaC_43',
    'PaviaC_51', 'PaviaC_52', 'PaviaC_53'
    ]

    names_Houston_list = [
    'Houston_11', 'Houston_21', 'Houston_31', 'Houston_41', 'Houston_51'
    ]

    names_dc_list = [
    'dc_11', 'dc_12', 'dc_21', 'dc_22',
    'dc_31', 'dc_32', 'dc_41', 'dc_42',
    'dc_51', 'dc_52', 'dc_61', 'dc_62',
    'dc_71', 'dc_72', 'dc_81', 'dc_82',
    ]

    names_PaviaU_list = [
    'PaviaU_11', 'PaviaU_12', 'PaviaU_13', 'PaviaU_21', 'PaviaU_22', 'PaviaU_23',
    'PaviaU_31', 'PaviaU_32', 'PaviaU_33', 'PaviaU_41', 'PaviaU_42', 'PaviaU_43',
    'PaviaU_51', 'PaviaU_52', 'PaviaU_53'
    ]

    names_Houston, names_Houston_valid = split(names_Houston_list, shuffle=True, ratio=0.6)
    names_dc, names_dc_valid = split(names_dc_list, shuffle=True, ratio=0.7)
    names_PaviaU, names_PaviaU_valid = split(names_PaviaU_list, shuffle=True, ratio=0.67)
    names_PaviaC, names_PaviaC_valid = split(names_PaviaC_list, shuffle=True, ratio=0.67)


    if datasets == 'PaviaC':
        names = names_PaviaC
    elif datasets == 'PaviaC_val':
        names = names_PaviaC_valid
    elif datasets == 'PaviaU':
        names = names_PaviaU
    elif datasets == 'PaviaU_val':
        names = names_PaviaU_valid
    elif datasets == 'Houston':
        names = names_Houston
    elif datasets == 'Houston_val':
        names = names_Houston_valid
    elif datasets == 'dc':
        names = names_dc
    elif datasets == 'dc_val':
        names = names_dc_valid

    else:
        assert 'wrong dataset name'
    return names
