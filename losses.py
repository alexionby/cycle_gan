import torch
import torchvision


# Initialize Loss function
criterion_Im = torch.nn.L1Loss()


def LSGAN_D(real, fake):
    return 0.5 * (torch.mean((real - 1)**2) + torch.mean(fake**2))


def LSGAN_G(fake):
    return torch.mean((fake - 1)**2)


# Feature matching loss
def FM_G(a, b):
    assert len(a) == len(b)
    loss = 0
    for i, j in zip(a, b):
        # loss += torch.mean(torch.abs(i - j))
        loss += torch.mean((i - j)**2)
    loss = loss / len(a)
    return loss


# def RaLSGAN_D(fake, real):
#     loss = (torch.mean((real - torch.mean(fake) - 1) ** 2) +
#             torch.mean((fake - torch.mean(real) + 1) ** 2))/2
#     return loss


# def RaLSGAN_G(fake, real):
#     loss = (torch.mean((real - torch.mean(fake) + 1) ** 2) +
#             torch.mean((fake - torch.mean(real) - 1) ** 2))/2
#     return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss