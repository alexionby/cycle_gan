import os
import numpy as np
import glob
import time
import PIL.Image as Image
from tqdm.notebook import tqdm
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


# from losses import VGGPerceptualLoss
from models import Discriminator, InceptionGenerator
from losses import LSGAN_G, LSGAN_D, FM_G, criterion_Im
from utils import (ReplayBuffer, weights_init,
                   save_images_test, save_models, load_models)
from dataloader import get_dataset


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)


# vgg_loss = VGGPerceptualLoss(resize=False)
# vgg_loss = vgg_loss.to("cuda")


args = parser.parse_args()


image_size = (128, 128)
buffer_size = 100
lr = 0.0002
beta1 = 0.5
epochs = 25
decay_epochs = 25

bs = 8
workers = 8
# device = 'cuda:1'

if args.local_rank == 0:
    writer = SummaryWriter('train_logs_ddp')

name = "smile_inception"  # experiment name

folder = '../../Projects/Meta/smile_ultimate'
trainA = f'{folder}/trainA'
trainB = f'{folder}/trainB'
testA = f'{folder}/valA'
testB = f'{folder}/valB'


torch.cuda.set_device(0)
world_size = 2

torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank
)

print(args.local_rank)


G_A2B = InceptionGenerator(image_size=image_size[0]) # .to(device)
G_B2A = InceptionGenerator(image_size=image_size[0])# .to(device)

D_A = Discriminator() #.to(device)
D_B = Discriminator() #.to(device)

D_A_x2 = Discriminator() #.to(device)
D_B_x2 = Discriminator() #.to(device)

D_A_x4 = Discriminator() #.to(device)
D_B_x4 = Discriminator() #.to(device)

G_A2B.apply(weights_init)
G_B2A.apply(weights_init)
D_A.apply(weights_init)
D_B.apply(weights_init)

D_A_x2.apply(weights_init)
D_B_x2.apply(weights_init)
D_A_x4.apply(weights_init)
D_B_x4.apply(weights_init)


G_A2B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G_A2B)
G_B2A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G_B2A)

D_A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D_A)
D_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D_B)

device = torch.device('cuda:{}'.format(args.local_rank))

G_A2B = G_A2B.to(device)
G_B2A = G_B2A.to(device)

D_A = D_A.to(device)
D_B = D_B.to(device)

G_A2B = torch.nn.parallel.DistributedDataParallel(
    G_A2B,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)

G_B2A = torch.nn.parallel.DistributedDataParallel(
    G_B2A,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)

D_A = torch.nn.parallel.DistributedDataParallel(
    D_A,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)

D_B = torch.nn.parallel.DistributedDataParallel(
    D_B,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)

trainA_ds = get_dataset(trainA, image_size)
trainB_ds = get_dataset(trainB, image_size)
testA_ds = get_dataset(testA, image_size)
testB_ds = get_dataset(testB, image_size)


sampler_trA = torch.utils.data.distributed.DistributedSampler(
    trainA_ds,
    num_replicas=2,
    rank=args.local_rank,
)

trainA_gen = DataLoader(trainA_ds, bs,
                        pin_memory=True,
                        sampler=sampler_trA,
                        drop_last=True,
                        num_workers=workers)


sampler_trB = torch.utils.data.distributed.DistributedSampler(
    trainB_ds,
    num_replicas=2,
    rank=args.local_rank,
)

trainB_gen = DataLoader(trainB_ds, bs,
                        pin_memory=True,
                        sampler=sampler_trB,
                        drop_last=True,
                        num_workers=workers)

sampler_teA = torch.utils.data.distributed.DistributedSampler(
    testA_ds,
    num_replicas=2,
    rank=args.local_rank,
)

testA_gen = DataLoader(testA_ds, bs, 
                        pin_memory=True,
                        sampler=sampler_teA,
                        drop_last=True,
                        num_workers=workers)

sampler_teB = torch.utils.data.distributed.DistributedSampler(
    testB_ds,
    num_replicas=2,
    rank=args.local_rank,
)

testB_gen = DataLoader(testB_ds, bs, 
                        pin_memory=True,
                        sampler=sampler_teA,
                        drop_last=True,
                        num_workers=workers)


# print("Generator summary: ")
# print(summary(G_A2B, (3, *image_size), device='cuda'))

# print("Discriminator summary: ")
# print(summary(D_A, (3, *image_size), device='cuda'))

# G_A2B, G_B2A, D_A, D_B = load_models(name)

# similar optimizer for G and D???

optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=lr, betas=(beta1, 0.999))

# try to put SGD here
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

optimizer_D_A_x2 = torch.optim.Adam(D_A_x2.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B_x2 = torch.optim.Adam(D_B_x2.parameters(), lr=lr, betas=(beta1, 0.999))

optimizer_D_A_x4 = torch.optim.Adam(D_A_x4.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B_x4 = torch.optim.Adam(D_B_x4.parameters(), lr=lr, betas=(beta1, 0.999))



def training(G_A2B, G_B2A, D_A, D_B, D_A_x2, D_B_x2, name):

    if 'logs' not in os.listdir():
        os.mkdir('logs')

    # Training Loop
    iters = 0

    fake_A_buffer = ReplayBuffer(max_size=buffer_size)
    fake_B_buffer = ReplayBuffer(max_size=buffer_size)

    fake_A_buffer_x2 = ReplayBuffer(max_size=buffer_size)
    fake_B_buffer_x2 = ReplayBuffer(max_size=buffer_size)

    fake_A_buffer_x4 = ReplayBuffer(max_size=buffer_size)
    fake_B_buffer_x4 = ReplayBuffer(max_size=buffer_size)

    lr_decay_epochs = np.linspace(lr, 0, decay_epochs)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(epochs + decay_epochs):

        if epoch > epochs:
            for optim in [optimizer_D_A, optimizer_D_B,
                          optimizer_G_A2B, optimizer_G_B2A,
                          optimizer_D_A_x2, optimizer_D_B_x2,
                          optimizer_D_A_x4, optimizer_D_B_x4]:
                for g in optim.param_groups:
                    g['lr'] = lr_decay_epochs[epoch - epochs]

        # print(epoch, optimizer_D_A.param_groups[0]['lr'])

        # For each batch in the dataloader
        for i, (data_horse, data_zebra) in enumerate(zip(trainA_gen, trainB_gen), 0):

            # set model input
            a_real, a_real_x2, a_real_x4 = [d.to(device) for d in data_horse]
            b_real, b_real_x2, b_real_x4 = [d.to(device) for d in data_zebra]

            # Generated images
            b_fake, b_fake_x2, b_fake_x4 = G_A2B(a_real)
            a_rec, a_rec_x2, a_rec_x4 = G_B2A(b_fake)
            a_fake, a_fake_x2, a_fake_x4 = G_B2A(b_real)
            b_rec, b_rec_x2, b_rec_x4 = G_A2B(a_fake)

            a_idt, _, _ = G_B2A(a_real)
            b_idt, _, _ = G_A2B(b_real)

            # Generator

            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()

            # Fool discriminator

            db_fake, db_feat = D_B(b_fake, True)
            _, db_feat_real = D_B(b_real, True)

            da_fake, da_feat = D_A(a_fake, True)
            _, da_feat_real = D_A(a_real, True)

            # print([d.shape for d in da_feat], [d.shape for d in da_feat_real])
            
            # db_fake_x2, db_feat_x2 = D_B_x2(b_fake_x2, True)
            # _, db_feat_real_x2 = D_B_x2(b_real_x2, True)

            # da_fake_x2, da_feat_x2 = D_A_x2(a_fake_x2, True)
            # _, da_feat_real_x2 = D_A_x2(a_real_x2, True)

            # print('Real: ', b_real_x2.shape, b_fake_x2.shape)
            # print([d.shape for d in da_feat_x2], [d.shape for d in da_feat_real_x2])

            # db_fake_x4, db_feat_x4 = D_B_x4(b_fake_x4, True)
            # _, db_feat_real_x4 = D_B_x4(b_real_x4, True)

            # da_fake_x4, da_feat_x4 = D_A_x4(a_fake_x4, True)
            # _, da_feat_real_x4 = D_A_x4(a_real_x4, True)

            # FM for A2B
            # FM_A2B_loss = 0
            # for a, b in [(db_feat, db_feat_real),
            #              (db_feat_x2, db_feat_real_x2),
            #              (db_feat_x4, db_feat_real_x4)]:
            #     FM_A2B_loss += FM_G(a, b)

            # FM for B2A
            # FM_B2A_loss = 0
            # for a, b in [(da_feat, da_feat_real),
            #              (da_feat_x2, da_feat_real_x2),
            #              (da_feat_x4, da_feat_real_x4)]:
            #     FM_B2A_loss += FM_G(a, b)

            Gen_loss_A2B = LSGAN_G(db_fake)
            Gen_loss_B2A = LSGAN_G(da_fake)

            # Gen_loss_A2B_x2 = LSGAN_G(db_fake_x2)
            # Gen_loss_B2A_x2 = LSGAN_G(da_fake_x2)
            # Gen_loss_A2B_x4 = LSGAN_G(db_fake_x4)
            # Gen_loss_B2A_x4 = LSGAN_G(da_fake_x4)

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
            Loss_G = Gen_loss_A2B + Gen_loss_B2A
            # Loss_G = Loss_G + Gen_loss_A2B_x2 + Gen_loss_B2A_x2
            # Loss_G = Loss_G + Gen_loss_A2B_x4 + Gen_loss_B2A_x4
            # Loss_G = Loss_G / 3.0
            # Loss_G = Loss_G + FM_A2B_loss + FM_B2A_loss
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
            # optimizer_D_A_x2.zero_grad()
            # optimizer_D_A_x4.zero_grad()

            buffer_a_fake = fake_A_buffer.push_and_pop(a_fake)
            # buffer_a_fake_x2 = fake_A_buffer_x2.push_and_pop(a_fake_x2)
            # buffer_a_fake_x4 = fake_A_buffer_x4.push_and_pop(a_fake_x4)

            Disc_loss_A = LSGAN_D(D_A(a_real), D_A(buffer_a_fake.detach()))
            # Disc_loss_A_x2 = LSGAN_D(D_A_x2(a_real_x2), D_A_x2(buffer_a_fake_x2.detach()))
            # Disc_loss_A_x4 = LSGAN_D(D_A_x4(a_real_x4), D_A_x4(buffer_a_fake_x4.detach()))

            Disc_loss_A.backward()
            optimizer_D_A.step()
            # Disc_loss_A_x2.backward()
            # optimizer_D_A_x2.step()
            # Disc_loss_A_x4.backward()
            # optimizer_D_A_x4.step()

            # Discriminator B
            optimizer_D_B.zero_grad()
            # optimizer_D_B_x2.zero_grad()
            # optimizer_D_B_x4.zero_grad()

            buffer_b_fake = fake_B_buffer.push_and_pop(b_fake)
            # buffer_b_fake_x2 = fake_B_buffer_x2.push_and_pop(b_fake_x2)
            # buffer_b_fake_x4 = fake_B_buffer_x4.push_and_pop(b_fake_x4)

            Disc_loss_B = LSGAN_D(D_B(b_real), D_B(buffer_b_fake.detach()))
            # Disc_loss_B_x2 = LSGAN_D(D_B_x2(b_real_x2), D_B_x2(buffer_b_fake_x2.detach()))
            # Disc_loss_B_x4 = LSGAN_D(D_B_x4(b_real_x4), D_B_x4(buffer_b_fake_x4.detach()))

            Disc_loss_B.backward()
            optimizer_D_B.step()
            # Disc_loss_B_x2.backward()
            # optimizer_D_B_x2.step()
            # Disc_loss_B_x4.backward()
            # optimizer_D_B_x4.step()

            if args.local_rank == 0:

                writer.add_scalars("Gen/Loss",
                                {"A": Gen_loss_A2B,
                                    "B": Gen_loss_B2A},
                                    # "Ax2": Gen_loss_A2B_x2,
                                    # "Bx2": Gen_loss_B2A_x2,
                                    # "Ax4": Gen_loss_A2B_x4,
                                    # "Bx4": Gen_loss_B2A_x4},
                                iters)

                writer.add_scalars("Disc/Loss",
                                {"A": Disc_loss_A,
                                    "B": Disc_loss_B},
                                    # "Ax2": Disc_loss_A_x2,
                                    # "Bx2": Disc_loss_B_x2,
                                    # "Ax4": Disc_loss_A_x4,
                                    # "Bx4": Disc_loss_B_x4},
                                iters)

                # writer.add_scalar('FM/A2B', FM_A2B_loss, iters)
                # writer.add_scalar('FM/B2A', FM_B2A_loss, iters)

                writer.add_scalar('Cycle/Cycle_A->B->A_loss', Cycle_loss_A, iters)
                writer.add_scalar('Cycle/Cycle_B->A->B_loss', Cycle_loss_B, iters)
                writer.add_scalar('Idt/Identity_A2B_L1', Id_loss_A2B, iters)
                writer.add_scalar('Idt/Identity_B2A_L1', Id_loss_B2A, iters)

            if iters % 100 == 0:
                print('[%d/%d] Iters: %d | G_A: %.4f | G_B: %.4f | Cycle_A: %.4f | Cycle_B: %.4f | Idt_B2A: %.4f | Idt_A2B: %.4f | Loss_D_A: %.4f | Loss_D_B: %.4f'
                        % (epoch+1, epochs+decay_epochs, iters, Gen_loss_A2B, Gen_loss_B2A,Cycle_loss_A,Cycle_loss_B,Id_loss_B2A,
                            Id_loss_A2B, Disc_loss_A, Disc_loss_B))

            if (iters % 100 == 0):
                if args.local_rank == 0:
                    pass
                    # save_images_test(testA_gen, testB_gen, G_A2B, G_B2A, epoch, device)

            iters += 1

        if args.local_rank == 0:
            print('Saving_models! Epoch: ', epoch)
            save_models(G_A2B, G_B2A, D_A, D_B, name)

        # save_models(G_A2B, G_B2A, D_A, D_B, D_A_x2, D_B_x2, D_A_x4, D_B_x4, name)


losses = training(G_A2B, G_B2A, D_A, D_B, D_A_x2, D_B_x2, name)
if args.local_rank == 0:
    writer.close()