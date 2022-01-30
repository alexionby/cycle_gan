import os
from cv2 import randShuffle
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
from tensorboardX import SummaryWriter


from losses import VGGPerceptualLoss


vgg_loss = VGGPerceptualLoss(resize=False)
vgg_loss = vgg_loss.to("cuda")


writer = SummaryWriter('train_logs')


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


image_size = (256, 256)

folder = 'face2bitmoji'
trainA = f'{folder}/trainA'
trainB = f'{folder}/trainB'
testA = f'{folder}/testA'
testB = f'{folder}/testB'


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.images = glob.glob(f"{img_dir}/*")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        pil_image = Image.open(img_path).convert("RGB")
        image = self.transform(pil_image)
        image_x2 = image.unsqueeze(0)
        image_x2 = F.interpolate(image_x2, size=128, mode='area')
        image_x2 = image_x2.squeeze(0)
        pil_image.close()
        return image, image_x2


transform = transforms.Compose([transforms.Resize(image_size, Image.LANCZOS),
                                transforms.RandomCrop(image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                # RandomResizedCrop,
                                ])


bs = 1
workers = 8
device = 'cuda:0'

from models import Generator, ResBlock, Discriminator, norm_layer


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


def LSGAN_D(real, fake):
    return 0.5 * (torch.mean((real - 1)**2) + torch.mean(fake**2))


def LSGAN_G(fake):
    return torch.mean((fake - 1)**2)



# def RaLSGAN_D(fake, real):
#     loss = (torch.mean((real - torch.mean(fake) - 1) ** 2) +
#             torch.mean((fake - torch.mean(real) + 1) ** 2))/2
#     return loss


# def RaLSGAN_G(fake, real):
#     loss = (torch.mean((real - torch.mean(fake) + 1) ** 2) +
#             torch.mean((fake - torch.mean(real) - 1) ** 2))/2
#     return loss


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
    
    with torch.no_grad():

        batch_a_test, batch_a_test_x2 = [g.to(device) for g in next(iter(genA))]
        # real_a_test = batch_a_test.cpu().detach()
        fake_b_test, fake_b_test_x2 = G_A2B(batch_a_test)
        real_a_inv, real_a_inv_x2 = [b.cpu() for b in G_B2A(fake_b_test)]
        fake_b_test = fake_b_test.cpu()
        fake_b_test_x2 = fake_b_test_x2.cpu()
        batch_a_test = batch_a_test.cpu()
        batch_a_test_x2 = batch_a_test_x2.cpu()

        # print(batch_a_test_x2.shape, batch_a_test.shape)

        batch_b_test = next(iter(genB))[0].to(device)
        # real_b_test = batch_b_test.cpu().detach()
        fake_a_test = G_B2A(batch_b_test)[0]
        real_b_inv = G_A2B(fake_a_test)[0].cpu()
        fake_a_test = fake_a_test.cpu()
        batch_b_test = batch_b_test.cpu()

        fig, ax = plt.subplots(2, 3, figsize=(10, 10))

        ax[0,0].imshow(np.transpose(vutils.make_grid((batch_a_test[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        ax[0,0].set_title("Real Image")
        # ax[0,0].axis('off')
        ax[0,1].imshow(np.transpose(vutils.make_grid((fake_b_test[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        ax[0,1].set_title("Fake Image")
        # ax[0,1].axis('off')
        ax[0,2].imshow(np.transpose(vutils.make_grid((real_a_inv[:4]+1)/2, padding=2, normalize=True),(1,2,0)))
        ax[0,2].set_title("Inversed Image")
        # ax[0,2].axis('off')

        ax[1,0].imshow(np.transpose(vutils.make_grid((batch_a_test_x2[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        ax[1,0].set_title("Real Image")
        # ax[1,0].axis('off')
        ax[1,1].imshow(np.transpose(vutils.make_grid((fake_b_test_x2[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        ax[1,1].set_title("Fake Image")
        # ax[1,1].axis('off')
        ax[1,2].imshow(np.transpose(vutils.make_grid((real_a_inv_x2[:4]+1)/2, padding=2, normalize=True),(1,2,0)))
        ax[1,2].set_title("Inversed Image")
        # ax[1,2].axis('off')

        fig.savefig(f"logs/ep_{epoch}_A->B->A.png")
        plt.close('all')

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        
        ax[0].imshow(np.transpose(vutils.make_grid((batch_b_test[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        ax[0].set_title("Real Image")
        ax[0].axis('off')
        ax[1].imshow(np.transpose(vutils.make_grid((fake_a_test[:4]+1)/2, padding=2, normalize=True),(1,2,0)))
        ax[1].set_title("Fake Image")
        ax[1].axis('off')
        ax[2].imshow(np.transpose(vutils.make_grid((real_b_inv[:4]+1)/2, padding=2, normalize=True),(1,2,0)))
        ax[2].set_title("Inversed Image")
        ax[2].axis('off')

        fig.savefig(f"logs/ep_{epoch}_B->A->B.png")
        plt.close('all')


G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

D_A_x2 = Discriminator().to(device)
D_B_x2 = Discriminator().to(device)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

G_A2B.apply(weights_init)
G_B2A.apply(weights_init)
D_A.apply(weights_init)
D_B.apply(weights_init)
D_A_x2.apply(weights_init)
D_B_x2.apply(weights_init)

# print("Generator summary: ")
# print(summary(G_A2B, (3, *image_size), device='cuda'))

# print("Discriminator summary: ")
# print(summary(D_A, (3, *image_size), device='cuda'))

# Initialize Loss function
criterion_Im = torch.nn.L1Loss()

# Buffer size
buffer_size = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
name = "horse2zebra_256"
epochs = 100
decay_epochs = 100

# G_A2B, G_B2A, D_A, D_B = load_models(name)

# similar optimizer for G and D???

optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr, betas=(beta1, 0.999))

# try to put SGD here
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

optimizer_D_A_x2 = torch.optim.Adam(D_A_x2.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B_x2 = torch.optim.Adam(D_B_x2.parameters(), lr=lr, betas=(beta1, 0.999))


def training(G_A2B, G_B2A, D_A, D_B, D_A_x2, D_B_x2, name):

    # Training Loop
    iters = 0

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    fake_A_buffer_x2 = ReplayBuffer()
    fake_B_buffer_x2 = ReplayBuffer()

    lr_decay_epochs = np.linspace(lr, 0, decay_epochs)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(epochs + decay_epochs):

        if epoch > epochs:
            for optim in [optimizer_D_A, optimizer_D_B, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A_x2, optimizer_D_B_x2]:
                for g in optim.param_groups:
                    g['lr'] = lr_decay_epochs[epoch - epochs]

        # print(epoch, optimizer_D_A.param_groups[0]['lr'])

        # For each batch in the dataloader
        for i, (data_horse, data_zebra) in enumerate(zip(trainA_gen, trainB_gen), 0):
            
            # set model input
            a_real, a_real_x2 = [d.to(device) for d in data_horse]
            b_real, b_real_x2 = [d.to(device) for d in data_zebra]

            # Generated images
            b_fake, b_fake_x2 = G_A2B(a_real)
            a_rec, a_rec_x2 = G_B2A(b_fake)
            a_fake, a_fake_x2 = G_B2A(b_real)
            b_rec, b_rec_x2 = G_A2B(a_fake)

            a_idt, _ = G_B2A(a_real)
            b_idt, _ = G_A2B(b_real)

            # Generator

            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()

            # Fool discriminator
            Gen_loss_A2B = LSGAN_G(D_B(b_fake))
            Gen_loss_B2A = LSGAN_G(D_A(a_fake))
            Gen_loss_A2B_x2 = LSGAN_G(D_B_x2(b_fake_x2))
            Gen_loss_B2A_x2 = LSGAN_G(D_A_x2(a_fake_x2))

            # Fool_disc_loss_A2B = RaLSGAN_G(D_B(b_fake), D_B(b_real))
            # Fool_disc_loss_B2A = RaLSGAN_G(D_A(a_fake), D_A(a_real))

            lambda_A = 10
            lambda_B = 10
            lambda_I = 0.5

            # Cycle Consistency    both use the two generators
            Cycle_loss_A = criterion_Im(a_rec, a_real)*lambda_A
            Cycle_loss_B = criterion_Im(b_rec, b_real)*lambda_B

            # Identity loss
            Id_loss_B2A = criterion_Im(a_idt, a_real)*lambda_A*lambda_I
            Id_loss_A2B = criterion_Im(b_idt, b_real)*lambda_B*lambda_I

            # VGG loss
            # VGG_A = vgg_loss(a_real, a_fake)
            # VGG_B = vgg_loss(b_fake, b_real)

            # generator losses
            Loss_G = Gen_loss_A2B + Gen_loss_A2B_x2 + Gen_loss_B2A + Gen_loss_B2A_x2
            Loss_G = Loss_G / 2.0
            Loss_G = Loss_G + Cycle_loss_A + Cycle_loss_B
            Loss_G = Loss_G + Id_loss_A2B + Id_loss_B2A
            # Loss_G = Loss_G + VGG_A + VGG_B

            # Backward propagation
            Loss_G.backward()

            # Optimisation step
            optimizer_G_A2B.step()
            optimizer_G_B2A.step()

            # if iters == 0:
            #     buffer_A = a_fake.detach()
            #     buffer_B = b_fake.detach()
            # else:
            #     buffer_A = torch.cat([a_fake.detach(), buffer_A])
            #     buffer_B = torch.cat([b_fake.detach(), buffer_B])

            #     if len(buffer_A) > buffer_size:
            #         buffer_A = buffer_A[:50]
            #         buffer_B = buffer_B[:50]

            #     idx_A = torch.randperm(buffer_A.shape[0])
            #     idx_B = torch.randperm(buffer_B.shape[0])

            #     idx_A = idx_A[:buffer_size]
            #     idx_B = idx_B[:buffer_size]

            #     buffer_A = buffer_A[idx_A].view(buffer_A.size())
            #     buffer_B = buffer_B[idx_B].view(buffer_B.size())

            # buffer_a_fake = buffer_A[:bs]
            # buffer_b_fake = buffer_B[:bs]

            # Discriminator A
            optimizer_D_A.zero_grad()
            optimizer_D_A_x2.zero_grad()

            buffer_a_fake = fake_A_buffer.push_and_pop(a_fake)
            buffer_a_fake_x2 = fake_A_buffer_x2.push_and_pop(a_fake_x2)

            Disc_loss_A = LSGAN_D(D_A(a_real), D_A(buffer_a_fake.detach()))
            Disc_loss_A_x2 = LSGAN_D(D_A_x2(a_real_x2), D_A_x2(buffer_a_fake_x2.detach()))
            # Disc_loss_A = RaLSGAN_D(D_A(a_real), D_A(buffer_a_fake.detach()))

            Disc_loss_A.backward()
            optimizer_D_A.step()

            Disc_loss_A_x2.backward()
            optimizer_D_A_x2.step()

            # Discriminator B
            optimizer_D_B.zero_grad()
            optimizer_D_B_x2.zero_grad()

            buffer_b_fake = fake_B_buffer.push_and_pop(b_fake)
            buffer_b_fake_x2 = fake_B_buffer_x2.push_and_pop(b_fake_x2)

            Disc_loss_B = LSGAN_D(D_B(b_real), D_B(buffer_b_fake.detach()))
            Disc_loss_B_x2 = LSGAN_D(D_B_x2(b_real_x2), D_B_x2(buffer_b_fake_x2.detach()))
            # Disc_loss_B = RaLSGAN_D(D_B(b_real), D_B(buffer_b_fake.detach()))

            Disc_loss_B.backward()
            optimizer_D_B.step()

            Disc_loss_B_x2.backward()
            optimizer_D_B_x2.step()

            writer.add_scalars("Gen/Loss",
                               {"A": Gen_loss_A2B,
                                "B": Gen_loss_B2A,
                                "Ax2": Gen_loss_A2B_x2,
                                "Bx2": Gen_loss_B2A_x2},
                               iters)

            writer.add_scalars("Disc/Loss",
                               {"A": Disc_loss_A,
                                "B": Disc_loss_B,
                                "Ax2": Disc_loss_A_x2,
                                "Bx2": Disc_loss_B_x2},
                               iters)

            writer.add_scalar('Cycle/Cycle_A->B->A_loss', Cycle_loss_A, iters)
            writer.add_scalar('Cycle/Cycle_B->A->B_loss', Cycle_loss_B, iters)
            writer.add_scalar('Idt/Identity_A2B_L1', Id_loss_A2B, iters)
            writer.add_scalar('Idt/Identity_B2A_L1', Id_loss_B2A, iters)

            # del data_zebra, data_horse, a_real, b_real, a_fake, b_fake
            # del buffer_a_fake, buffer_b_fake

            if iters % 100 == 0:
                print('[%d/%d] Iters: %d | G_A: %.4f | G_B: %.4f | Cycle_A: %.4f | Cycle_B: %.4f | Idt_B2A: %.4f | Idt_A2B: %.4f | Loss_D_A: %.4f | Loss_D_B: %.4f'
                        % (epoch+1, epochs+decay_epochs, iters, Gen_loss_A2B, Gen_loss_B2A,Cycle_loss_A,Cycle_loss_B,Id_loss_B2A,
                            Id_loss_A2B, Disc_loss_A, Disc_loss_B))

            if (iters % 100 == 0):
                save_images_test(testA_gen, testB_gen, epoch)

            iters += 1

        save_models(G_A2B, G_B2A, D_A, D_B, name)


losses = training(G_A2B, G_B2A, D_A, D_B, D_A_x2, D_B_x2, name)
writer.close()
