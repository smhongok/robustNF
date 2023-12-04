import os
import glob
import cv2
import math
import argparse
from math import log10
import numpy as np
from tqdm.autonotebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
# import albumentations.pytorch
# import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision.models.vgg import vgg19
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torchvision.utils as utils

from models import create_model
import options.options as option
from utils.util import opt_get
import csv


def t(array): return torch.Tensor((array.transpose([2, 0, 1])).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def random_crop(hr, lr, size_hr, scale, random):
    size_lr = size_hr // scale
    
    size_lr_x = lr.shape[1]
    size_lr_y = lr.shape[0]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_y_lr:start_y_lr + size_lr, start_x_lr:start_x_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr * scale
    start_y_hr = start_y_lr * scale
    hr_patch = hr[start_y_hr:start_y_hr + size_hr, start_x_hr:start_x_hr + size_hr, :]

    return hr_patch, lr_patch


def center_crop(img, size):
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, border:-border, border:-border]

class HRLRDatasetV(Dataset):
    def __init__(self, lr_list):
        super(HRLRDatasetV, self).__init__()
        self.lr = lr_list

    def __getitem__(self, index):
        lr_path = self.lr[index]
        lr = imread(lr_path)
        #print(lr.shape)
        pad_factor = 2
        # Pad image to be % 2
        
        h, w, c = lr.shape
        lq_orig = lr.copy()
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))

        lr_t = t(lr)
        return lr_t, h, w
    def __len__(self):
        return len(self.lr)

def load_model(conf_path, keys, values, _resume=None):
    opt = option.parse(conf_path, keys, values, is_train=False)
    opt['gpu_ids'] = 0
    opt = option.dict_to_nonedict(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['gpu_ids'])
    model = create_model(opt)
    model_path = opt_get(opt, ['model_path'], None) \
        if _resume is None else os.path.join('./experiments', opt_get(opt, ['name']), 'models', _resume + '_G.pth')
    print(model_path)
    model.load_network(load_path=model_path, network=model.netG, strict=True)
    return model, opt

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--heat', type=float, default=0.9)
    parser.add_argument('--n_sample', type=int, default=10)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--lrtest_path', default='../data/DIV2K-va_4X', type=str) # lr test
    parser.add_argument('--conf_path', default='./confs/4X.yml', type=str)
    parser.add_argument('--output_path', default="./experiments/4X_15/", type=str) # output path
    parser.add_argument('--n_imgs', type=int, default=100)
    parser.add_argument('--override', default="")
    parser.add_argument('--resume-ckpt')
    args = parser.parse_args()

    keys, values = [], []
    for op in args.override.split('|'):
        if not op:
            continue
        address, value = op.split('=')
        key = address.split('.')
        keys.append(key)
        values.append(value)

    SEED = args.seed
    heat = args.heat
    n_sample = args.n_sample
    scale = args.scale
    lrtest_path = args.lrtest_path
    conf_path = args.conf_path
    test_root = args.output_path
    n_imgs = args.n_imgs

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if os.path.isdir(test_root) is False: os.mkdir(test_root)  
    if torch.cuda.is_available(): 
        device = 'cuda'
    else: 
        device = 'cpu'

    lr_test = sorted(glob.glob(lrtest_path + '/*'))
    print((lrtest_path + '/*'))
    validset = HRLRDatasetV(lr_test)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)

    model, opt = load_model(conf_path, keys, values, args.resume_ckpt)
    #print(model)
    model.to(device)
    print(device)

    ##### IMAGE SAVE #####
    progress_bar = tqdm(validloader)
    check = list(range(0,n_imgs))
    #check = [6,20,27,46,99,0,1,2,3,4]
    fields = ['i', 'ii', 'errpixel (1.5,-0.5)', 'nanpixel']
    rows = []
    for i, data in enumerate(progress_bar):

        if i in check:
            with torch.no_grad():
                lr, h, w = data
                lr = lr.to(device)
                lr_upsample = torch.nn.functional.upsample(lr, scale_factor=scale, mode='bicubic')
                for ii in range(n_sample):
                    sr_t = model.get_sr(lq=lr, heat=heat)
                    sr_t = sr_t + lr_upsample
                    
                    #print(int(torch.any(sr_t > 1.5).item() or torch.any(sr_t <-0.5).item() ))
                    if torch.any(sr_t > 1.5).item() or torch.any(sr_t <-0.5).item() or torch.any(~torch.isfinite(sr_t)).item():
                        rows.append([i,ii,int(torch.any(sr_t > 1.5).item() or torch.any(sr_t<-0.5).item()),int(torch.any(~torch.isfinite(sr_t)).item())])
                    sr = rgb(torch.clamp(sr_t, 0, 1))
                    sr = sr[:h * scale, :w * scale]
                    path_out_sr = os.path.join(test_root, f'{i:06d}_sample{ii:05d}.png')
                    imwrite(path_out_sr, sr)

                    # if ii <10:
                    #     path_out_sr = test_root + f'{i:06d}_sample{ii:05d}.png'
                    # else:
                    #     new_ii = ii - 10
                    #     path_out_sr = tmp_root + f'{i:06d}_sample{new_ii:05d}.png'
                    # imwrite(path_out_sr, sr)
                    # #print(sr.shape) 

    with open(test_root+'/errorpixels.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)
