from email.mime import image
import os
import numpy as np
import glob
import time
import PIL.Image as Image
from tqdm.notebook import tqdm
from itertools import chain
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torchvision.utils as vutils

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import ipywidgets
from IPython import display
from torchsummary import summary

from models import Generator, ResBlock, norm_layer


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.images = glob.glob(f"{img_dir}/*")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

image_size = (256, 256)

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                # transforms.RandomHorizontalFlip(p=0.5),
                                # RandomResizedCrop,
                                ])


bs = 4
workers = 8
device = 'cuda:1'
nc = 3
ndf = 64
norm_layer = nn.InstanceNorm2d


def get_dataloader(dataroot, shuffle):
    dataset = ImageDataset(dataroot, transform)
    dataloader = DataLoader(dataset, bs, shuffle,
                            num_workers=workers)
    return dataloader


testA = 'horse2zebra/testA'
testB = 'horse2zebra/testB'

testA_gen = get_dataloader(testA, False)
testB_gen = get_dataloader(testB, False)


def load_models(name):
    G_A2B = torch.load("weights/"+name+"_G_A2B.pt")
    G_B2A = torch.load("weights/"+name+"_G_B2A.pt")
    return G_A2B, G_B2A

name = "horse2zebra_256"

G_A2B, G_B2A = load_models(name)

G_A2B.eval()
G_B2A.eval()

try:
    os.mkdir(f"result/{name}")
    os.mkdir(f"result/{name}/A")
    os.mkdir(f"result/{name}/B")
except:
    pass


def save_results(real, fake, rec, name, direction, idx):

    # image_orig = np.uint8(255 * (real[j].cpu()/2 + 0.5))
    # image_orig = np.transpose(image_orig, [1, 2, 0])
    # Image.fromarray(image_orig).save(f"result/{name}/{direction}/{idx}_real.png")

    image_fake = np.uint8(255 * (fake[j].cpu()/2 + 0.5))
    image_fake = np.transpose(image_fake, [1, 2, 0])
    Image.fromarray(image_fake).save(f"result/{name}/{direction}/{idx}_fake.png")

    # image_rec = np.uint8(255 * (rec[j].cpu()/2 + 0.5))
    # image_rec = np.transpose(image_rec, [1, 2, 0])
    # Image.fromarray(image_rec).save(f"result/{name}/{direction}/{idx}_rec.png")


idx = 0

for i, (data_A, data_B) in enumerate(zip(testA_gen, testB_gen), 0):

    a_real = data_A.to(device)
    b_real = data_B.to(device)

    with torch.no_grad():
        b_fake = G_A2B(a_real)
        a_rec = G_B2A(b_fake)
        a_fake = G_B2A(b_real)
        b_rec = G_A2B(a_fake)

    for j in range(a_real.shape[0]):
        idx = i*bs + j
        save_results(a_real, b_fake, a_rec, name, "A", idx)
        save_results(b_real, a_fake, b_rec, name, "B", idx)

# Input to the model
x = torch.randn(1, 3, 256, 256, requires_grad=True)
x = x.to(device)
torch_out = G_A2B(x)

# Export the model
torch.onnx.export(G_A2B,x,"h2z.onnx", opset_version=10)