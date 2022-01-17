from cgi import test
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


bs = 4
workers = 8
image_size = (256,256)

trainA = 'horse2zebra/trainA'
trainB = 'horse2zebra/trainB'
testA = 'horse2zebra/testA'
testB = 'horse2zebra/testB'

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


transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


bs = 4
workers = 8
device = 'cuda:0'
nc = 3
ndf = 64
norm_layer = nn.InstanceNorm2d


def get_dataloader(dataroot, shuffle):
    dataset = ImageDataset(dataroot, transform)
    dataloader = DataLoader(dataset, bs, shuffle,
                            num_workers=workers)
    return dataloader


trainA_gen = get_dataloader(trainA, True)
trainB_gen = get_dataloader(trainB, True)
testA_gen = get_dataloader(testA, True)
testB_gen = get_dataloader(testB, True)


# real_batch = next(iter(trainA_gen))
# print(real_batch.shape, real_batch.min(), real_batch.max())
# plt.figure(figsize=(12,4))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:10], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

# testA_batch = next(iter(testA_gen))
# print(testA_batch.shape, testA_batch.min(), testA_batch.max())
# plt.figure(figsize=(12,4))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(testA_batch.to(device)[:10], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), norm_layer(f), nn.ReLU(),
                                  nn.Conv2d(f, f, 3, 1, 1))
        self.norm = norm_layer(f)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))


class Generator(nn.Module):
    def __init__(self, f=64, blocks=6):
        super(Generator, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(  3,   f, 7, 1, 0), norm_layer(  f), nn.ReLU(True),
                  nn.Conv2d(  f, 2*f, 3, 2, 1), norm_layer(2*f), nn.ReLU(True),
                  nn.Conv2d(2*f, 4*f, 3, 2, 1), norm_layer(4*f), nn.ReLU(True)]
        for i in range(int(blocks)):
            layers.append(ResBlock(4*f))
        layers.extend([
                # nn.ConvTranspose2d(4*f, 4*2*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(2*f), nn.ReLU(True),
                # nn.ConvTranspose2d(2*f,   4*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(  f), nn.ReLU(True),

                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(4*f, 2*f, kernel_size=3, stride=1, padding=0),
                norm_layer(2*f),
                nn.ReLU(True),

                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(2*f, 1*f, kernel_size=3, stride=1, padding=0),
                norm_layer(f),
                nn.ReLU(True),

                nn.ReflectionPad2d(3), nn.Conv2d(f, 3, 7, 1, 0),
                nn.Tanh()])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):  
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc,ndf,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf,ndf*2,4,2,1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf*4,ndf*8,4,1,1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 15 x 15
            nn.Conv2d(ndf*8,1,4,1,1)
            # state size. 1 x 14 x 14
        )

    def forward(self, input):
        return self.main(input)


def LSGAN_D(real, fake):
    return 0.5 * (torch.mean((real - 1)**2) + torch.mean(fake**2))


def LSGAN_G(fake):
    return torch.mean((fake - 1)**2)


def save_models(G_A2B, G_B2A, D_A, D_B, name):
    torch.save(G_A2B, "weights/"+name+"_G_A2B.pt")
    torch.save(G_B2A, "weights/"+name+"_G_B2A.pt")
    torch.save(D_A, "weights/"+name+"_D_A.pt")
    torch.save(D_B, "weights/"+name+"_D_B.pt")


def load_models(name):
    G_A2B = torch.load("weights/"+name+"_G_A2B.pt")
    G_B2A = torch.load("weights/"+name+"_G_B2A.pt")
    D_A = torch.load("weights/"+name+"_D_A.pt")
    D_B = torch.load("weights/"+name+"_D_B.pt")
    return G_A2B, G_B2A, D_A, D_B


def save_images_test(genA, genB, epoch):
    batch_a_test = next(iter(genA)).to(device)
    real_a_test = batch_a_test.cpu().detach()
    fake_b_test = G_A2B(batch_a_test).cpu().detach()

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(vutils.make_grid((real_a_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Real horses")
    plt.savefig(f"logs/real_horses_iter_{epoch}.png")

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(vutils.make_grid((fake_b_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Fake zebras")
    plt.savefig(f"logs/fake_zebras_iter_{epoch}.png")

    batch_b_test = next(iter(genB)).to(device)
    real_b_test = batch_b_test.cpu().detach()
    fake_a_test = G_B2A(batch_b_test).cpu().detach()

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(vutils.make_grid((real_b_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Real zebras")
    plt.savefig(f"logs/real_zebras_iter_{epoch}.png")

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(vutils.make_grid((fake_a_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Fake horses")
    plt.savefig(f"logs/fake_horses_iter_{epoch}.png")

    plt.close('all')


G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

print("Generator summary: ")
print(summary(G_A2B, (3, 256, 256), device='cuda'))

print("Discriminator summary: ")
print(summary(D_A, (3, 256, 256), device='cuda'))

# Initialize Loss function
criterion_Im = torch.nn.L1Loss()

# Buffer size
buffer_size = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
name = "test_name"
epochs = 50

# G_A2B, G_B2A, D_A, D_B = load_models(name)
netG_A2B, netG_B2A, netD_A, netD_B = G_A2B, G_B2A, D_A, D_B

# similar optimizer for G and D???

optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr, betas=(beta1, 0.999))

# try to put SGD here
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))


def training(G_A2B, G_B2A, D_A, D_B, num_epochs, name):

    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_A_losses = []
    D_B_losses = []

    iters = 0
    FDL_A2B = []
    FDL_B2A = []
    CL_A = []
    CL_B = []
    ID_B2A = []
    ID_A2B = []
    disc_A = []
    disc_B = []

    FDL_A2B_t = []
    FDL_B2A_t = []
    CL_A_t = []
    CL_B_t = []
    ID_B2A_t = []
    ID_A2B_t = []
    disc_A_t = []
    disc_B_t = []

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data_horse, data_zebra) in enumerate(zip(trainA_gen, trainB_gen), 0):
        
            # Set model input
            a_real = data_horse.to(device)
            b_real = data_zebra.to(device)
        
            # Generated images
            b_fake = G_A2B(a_real)
            a_rec = G_B2A(b_fake)
            a_fake = G_B2A(b_real)
            b_rec = G_A2B(a_fake)

            if iters == 0:
                buffer_A = a_fake.clone()
                buffer_B = b_fake.clone()
            else:
                buffer_A = torch.cat([a_fake.clone(), buffer_A])
                buffer_B = torch.cat([b_fake.clone(), buffer_B])

                if len(buffer_A) > buffer_size:
                    buffer_A = buffer_A[:50]
                    buffer_B = buffer_B[:50]

                idx_A = torch.randperm(buffer_A.shape[0])
                idx_B = torch.randperm(buffer_B.shape[0])

                idx_A = idx_A[:buffer_size]
                idx_B = idx_B[:buffer_size]

                buffer_A = buffer_A[idx_A].view(buffer_A.size())
                buffer_B = buffer_B[idx_B].view(buffer_B.size())

            buffer_a_fake = buffer_A[:bs]
            buffer_b_fake = buffer_B[:bs]

            # print(len(buffer_A), len(buffer_B), a_fake.shape, b_fake.shape)

            # Discriminator A
            optimizer_D_A.zero_grad()
            Disc_loss_A = LSGAN_D(D_A(a_real), D_A(buffer_a_fake.detach()))
            D_A_losses.append(Disc_loss_A.item())
            Disc_loss_A.backward()
            optimizer_D_A.step()

            # Discriminator B
            optimizer_D_B.zero_grad()
            Disc_loss_B = LSGAN_D(D_B(b_real), D_B(buffer_b_fake.detach()))
            D_B_losses.append(Disc_loss_B.item())
            Disc_loss_B.backward()
            optimizer_D_B.step()

            # Generator

            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()

            # Fool discriminator
            Fool_disc_loss_A2B = LSGAN_G(D_B(b_fake))
            Fool_disc_loss_B2A = LSGAN_G(D_A(a_fake))

            lambda_A = 10
            lambda_B = 10
            lambda_I = 0.5

            # Cycle Consistency    both use the two generators
            Cycle_loss_A = criterion_Im(a_rec, a_real)*lambda_A
            Cycle_loss_B = criterion_Im(b_rec, b_real)*lambda_B

            # Identity loss
            Id_loss_B2A = criterion_Im(G_B2A(a_real), a_real)*lambda_A*lambda_I
            Id_loss_A2B = criterion_Im(G_A2B(b_real), b_real)*lambda_B*lambda_I

            # generator losses
            Loss_G = Fool_disc_loss_A2B + Fool_disc_loss_B2A
            Loss_G = Loss_G + Cycle_loss_A + Cycle_loss_B
            Loss_G = Loss_G + Id_loss_A2B + Id_loss_B2A
            G_losses.append(Loss_G)

            # Backward propagation
            Loss_G.backward()

            # Optimisation step
            optimizer_G_A2B.step()
            optimizer_G_B2A.step()

            FDL_A2B.append(Fool_disc_loss_A2B)
            FDL_B2A.append(Fool_disc_loss_B2A)
            CL_A.append(Cycle_loss_A)
            CL_B.append(Cycle_loss_B)
            ID_B2A.append(Id_loss_B2A)
            ID_A2B.append(Id_loss_A2B)
            disc_A.append(Disc_loss_A)
            disc_B.append(Disc_loss_B)

            iters += 1
            del data_zebra, data_horse, a_real, b_real, a_fake, b_fake

            if iters % 100 == 0:
                print('[%d/%d]\tIters: %d\tFDL_A2B: %.4f\tFDL_B2A: %.4f\tCL_A: %.4f\tCL_B: %.4f\tID_B2A: %.4f\tID_A2B: %.4f\tLoss_D_A: %.4f\tLoss_D_B: %.4f'
                        % (epoch+1, num_epochs, iters, Fool_disc_loss_A2B, Fool_disc_loss_B2A,Cycle_loss_A,Cycle_loss_B,Id_loss_B2A,
                            Id_loss_A2B, Disc_loss_A.item(), Disc_loss_B.item()))
            
            if (iters % 100 == 0):
                save_images_test(testA_gen, testB_gen, epoch)

        FDL_A2B_t.append(sum(FDL_A2B)/len(FDL_A2B))
        FDL_B2A_t.append(sum(FDL_B2A)/len(FDL_B2A))
        CL_A_t.append(sum(CL_A)/len(CL_A))
        CL_B_t.append(sum(CL_B)/len(CL_B))
        ID_B2A_t.append(sum(ID_B2A)/len(ID_B2A))
        ID_A2B_t.append(sum(ID_A2B)/len(ID_A2B))
        disc_A_t.append(sum(disc_A)/len(disc_A))
        disc_B_t.append(sum(disc_B)/len(disc_B))

        FDL_A2B = []
        FDL_B2A = []
        CL_A = []
        CL_B = []
        ID_B2A = []
        ID_A2B = []
        disc_B = []
        disc_A = []

        save_models(G_A2B, G_B2A, D_A, D_B, name)

    return (FDL_A2B_t, FDL_B2A_t,
            CL_A_t, CL_B_t, ID_B2A_t,
            ID_A2B_t, disc_A_t, disc_B_t)


losses = training(netG_A2B, netG_B2A, netD_A, netD_B, epochs, name)