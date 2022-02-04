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
from models import InceptionGenerator


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

# image_size = (256, 256)
image_size = (128, 128)

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


# testA = 'horse2zebra/testA'
# testB = 'horse2zebra/testB'

testA = '../../Projects/Meta/smile_ultimate/valA'
testB = '../../Projects/Meta/smile_ultimate/valB'


testA_gen = get_dataloader(testA, False)
testB_gen = get_dataloader(testB, False)


def load_models(name):
    G_A2B = InceptionGenerator(image_size=128)
    G_A2B.load_state_dict(torch.load("weights/"+name+"_G_A2B.pth", map_location=device))
    # G_A2B.load_state_dict(torch.load("weights.pth", map_location=device))

    G_B2A = InceptionGenerator(image_size=128)
    G_B2A.load_state_dict(torch.load("weights/"+name+"_G_B2A.pth", map_location=device), strict=False)
    return G_A2B, G_B2A

# name = "horse2zebra_256"
name = "smile_inception"

G_A2B, G_B2A = load_models(name)

G_A2B = G_A2B.to(device)
G_B2A = G_B2A.to(device)

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

    image_fake = np.uint8(255 * (fake.cpu()/2 + 0.5))
    image_fake = np.transpose(image_fake, [1, 2, 0])
    image_fake = Image.fromarray(image_fake)

    image_real = np.uint8(255 * (real.cpu()/2 + 0.5))
    image_real = np.transpose(image_real, [1, 2, 0])
    image_real = Image.fromarray(image_real)

    w,h = image_fake.size
    empty = Image.new('RGB', (w*2,h))
    empty.paste(image_real, (0,0))
    empty.paste(image_fake, (w,0))

    empty.save(f"result/{name}/{direction}/{idx}_fake.png")

    # image_rec = np.uint8(255 * (rec[j].cpu()/2 + 0.5))
    # image_rec = np.transpose(image_rec, [1, 2, 0])
    # Image.fromarray(image_rec).save(f"result/{name}/{direction}/{idx}_rec.png")


idx = 0

for i, (data_A, data_B) in enumerate(zip(testA_gen, testB_gen), 0):

    a_real = data_A.to(device)
    b_real = data_B.to(device)

    with torch.no_grad():
        b_fake, _, _ = G_A2B(a_real)
        a_rec, _, _ = G_B2A(b_fake)
        a_fake, _, _ = G_B2A(b_real)
        b_rec, _, _ = G_A2B(a_fake)

    for j in range(a_real.shape[0]):
        idx = i*bs + j
        save_results(a_real[j], b_fake[j], a_rec[j], name, "A", idx)
        save_results(b_real[j], a_fake[j], b_rec[j], name, "B", idx)

# Input to the model
# x = torch.randn(1, 3, 256, 256, requires_grad=True)
# x = x.to(device)
# torch_out = G_A2B(x)

# # Export the model
# torch.onnx.export(G_A2B,x,"h2z.onnx", opset_version=10)
