import torch
import random
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils


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


def weights_init(m):
    init_gain = 0.02
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.normal_(m.weight.data, 0.0, init_gain)

        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
        if hasattr(m, 'bias') and m.weight is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


# # custom weights initialization called on netG and netD
# def weights_init(m):
#     # print( type(m), isinstance(m, ConvNormRelu))
#     # if isinstance(m, ConvNormRelu):
#     #     print(sorted(dir(m)))
#     #     print(print(c) for c in m.named_children())
#     #     for i, j in enumerate(m.modules()):
#     #         print(i, type(j))
#     #         print(i, isinstance(j, nn.Module))
#     #         print(isinstance(j, nn.Sequential))
#     #         print(isinstance(j, ConvNormRelu))
#     #         print(i, j)

#     #         if not isinstance(j, ConvNormRelu) and not isinstance(j, nn.Sequential):
#     #             j.apply(weights_init)

#     classname = m.__class__.__name__
#     if hasattr(m, 'weight') and classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight, 0.0, 0.02)
#     elif classname.find("BatchNorm") != -1:
#         torch.nn.init.normal_(m.weight, 1.0, 0.02)
#         torch.nn.init.zeros_(m.bias)


def save_images_test(genA, genB, Gan_A2B, Gan_B2A, epoch, device):
    
    with torch.no_grad():
        pass

        batch_a_test, batch_a_test_x2, batch_a_test_x4 = [g.to(device) for g in next(iter(genA))]
        fake_b_test, fake_b_test_x2, fake_b_test_x4 = Gan_A2B(batch_a_test)
        # real_a_inv, real_a_inv_x2, real_a_inv_x4 = [b.cpu() for b in Gan_B2A(fake_b_test)]
        # fake_b_test = fake_b_test.cpu()
        # fake_b_test_x2 = fake_b_test_x2.cpu()
        # batch_a_test = batch_a_test.cpu()
        # batch_a_test_x2 = batch_a_test_x2.cpu()

        # print(batch_a_test.shape)
        # print(batch_a_test_x2.shape)
        # print(batch_a_test_x4.shape)
        # print(fake_b_test.shape)
        # print(fake_b_test_x2.shape)
        # print(fake_b_test_x4.shape)

        # batch_b_test = next(iter(genB))[0].to(device)
        # # real_b_test = batch_b_test.cpu().detach()
        # fake_a_test = Gan_B2A(batch_b_test)[0]
        # real_b_inv = Gan_A2B(fake_a_test)[0].cpu()
        # fake_a_test = fake_a_test.cpu()
        # batch_b_test = batch_b_test.cpu()

        # fig, ax = plt.subplots(3, 1, figsize=(10, 10))

        # ax[0].imshow(np.transpose(vutils.make_grid((batch_a_test[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        # ax[0].set_title("Real Image")
        # ax[0].axis('off')
        # ax[1].imshow(np.transpose(vutils.make_grid((fake_b_test[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        # ax[1].set_title("Fake Image")
        # ax[1].axis('off')
        # ax[2].imshow(np.transpose(vutils.make_grid((real_a_inv[:4]+1)/2, padding=2, normalize=True),(1,2,0)))
        # ax[2].set_title("Inversed Image")
        # ax[2].axis('off')
        # plt.tight_layout()

        # fig.savefig(f"logs/ep_{epoch}_A->B->A.png")
        # plt.close('all')

        # fig, ax = plt.subplots(3, 1, figsize=(10, 10))

        # ax[0].imshow(np.transpose(vutils.make_grid((batch_b_test[:4]+1)/2, padding=0, normalize=True),(1,2,0)))
        # ax[0].set_title("Real Image")
        # ax[0].axis('off')
        # ax[1].imshow(np.transpose(vutils.make_grid((fake_a_test[:4]+1)/2, padding=2, normalize=True),(1,2,0)))
        # ax[1].set_title("Fake Image")
        # ax[1].axis('off')
        # ax[2].imshow(np.transpose(vutils.make_grid((real_b_inv[:4]+1)/2, padding=2, normalize=True),(1,2,0)))
        # ax[2].set_title("Inversed Image")
        # ax[2].axis('off')

        # plt.tight_layout()

        # fig.savefig(f"logs/ep_{epoch}_B->A->B.png")
        # plt.close('all')


# def save_models(G_A2B, G_B2A, D_A, D_B, D_A_x2, D_B_x2, D_A_x4, D_B_x4, name):
#     torch.save(G_A2B, "weights/"+name+"_G_A2B.pt")
#     torch.save(G_B2A, "weights/"+name+"_G_B2A.pt")
#     torch.save(D_A, "weights/"+name+"_D_A.pt")
#     torch.save(D_B, "weights/"+name+"_D_B.pt")
#     torch.save(D_A_x2, "weights/"+name+"_D_A_x2.pt")
#     torch.save(D_B_x2, "weights/"+name+"_D_B_x2.pt")
#     torch.save(D_A_x4, "weights/"+name+"_D_A_x4.pt")
#     torch.save(D_B_x4, "weights/"+name+"_D_B_x4.pt")


def save_models(G_A2B, G_B2A, D_A, D_B, name):
    torch.save(G_A2B.module.state_dict(), f"weights/{name}_G_A2B.pth")
    torch.save(G_B2A.module.state_dict(), f"weights/{name}_G_B2A.pth")
    torch.save(D_A.module.state_dict(), f"weights/{name}_D_A2B.pth")
    torch.save(D_B.module.state_dict(), f"weights/{name}_D_B2A.pth")



def load_models(name):
    G_A2B = torch.load("weights/"+name+"_G_A2B.pt")
    G_B2A = torch.load("weights/"+name+"_G_B2A.pt")
    D_A = torch.load("weights/"+name+"_D_A.pt")
    D_B = torch.load("weights/"+name+"_D_B.pt")
    return G_A2B, G_B2A, D_A, D_B
