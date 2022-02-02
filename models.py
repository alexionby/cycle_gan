import torch
import torch.nn as nn
import torch.nn.functional as F

nc = 3
ndf = 64
norm_layer = nn.InstanceNorm2d


# class ResBlock(nn.Module):
#     def __init__(self, f):
#         super(ResBlock, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), norm_layer(f),
#                                   nn.ReLU(True),
#                                   nn.Conv2d(f, f, 3, 1, 1))
#         self.norm = norm_layer(f)

#     def forward(self, x):
#         return F.relu(self.norm(self.conv(x)+x))


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1),
                                  norm_layer(f),
                                  nn.ReLU(True),
                                  nn.Conv2d(f, f, 3, 1, 1))

        self.norm = norm_layer(f)

    def forward(self, x):
        return x + self.norm(self.conv(x))


class Generator(nn.Module):
    def __init__(self, f=64, blocks=9, image_size=256):
        super(Generator, self).__init__()

        # pad1 = nn.ReflectionPad2d(3)
        # enc1 = nn.Conv2d(  3,   f, 7, 1, 0), norm_layer(  f), nn.ReLU(True)

        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(  3,   f, 7, 1, 0), norm_layer(  f), nn.ReLU(True),
                  nn.Conv2d(  f, 2*f, 3, 2, 1), norm_layer(2*f), nn.ReLU(True),
                  nn.Conv2d(2*f, 4*f, 3, 2, 1), norm_layer(4*f), nn.ReLU(True)]

        for i in range(int(blocks)):
            layers.append(ResBlock(4*f))

        layers.extend([
                # option 1
                # nn.ConvTranspose2d(4*f, 4*2*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(2*f), nn.ReLU(True),
                # nn.ConvTranspose2d(2*f,   4*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(  f), nn.ReLU(True),

                # option 2
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                # nn.ReflectionPad2d(1),
                # nn.Conv2d(4*f, 2*f, kernel_size=3, stride=1, padding=0),
                # norm_layer(2*f),
                # nn.ReLU(True),

                # nn.Upsample(scale_factor=2, mode='bilinear'),
                # nn.ReflectionPad2d(1),
                # nn.Conv2d(2*f, 1*f, kernel_size=3, stride=1, padding=0),
                # norm_layer(f),
                # nn.ReLU(True),

                # option 3
                
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),

                nn.ReflectionPad2d(3), 
                nn.Conv2d(f, 3, 7, 1, 0),
                nn.Tanh()])

        self.conv = nn.Sequential(*layers)
        self.down_x2 = nn.Upsample(size=int(image_size/2))
        self.down_x4 = nn.Upsample(size=int(image_size/4))
        
    def forward(self, x):
        
        out = self.conv(x)
        out_down_x2 = self.down_x2(out)
        out_down_x4 = self.down_x4(out)

        return out, out_down_x2, out_down_x4


class Discriminator(nn.Module):  
    def __init__(self):
        super(Discriminator, self).__init__()

        down_blocks = []

        down_blocks.append(
            DownBlock2d(nc, ndf, 4, 2, 1)
        )

        down_blocks.append(
            DownBlock2d(ndf, ndf*2, 4, 2, 1, True)
        )

        down_blocks.append(
            DownBlock2d(ndf*2, ndf*4, 4, 2, 1, True)
        )

        down_blocks.append(
            DownBlock2d(ndf*4, ndf*8, 4, 1, 1, True)
        )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.final_conv = nn.Conv2d(ndf*8, out_channels=1, kernel_size=4, stride=1, padding=1)
        # 128 -> 1 x 14 x 14 / 256 -> 1 x 30 x 30

    def forward(self, x, features=False):

        feature_maps = []
        out = x

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]

        prediction_map = self.final_conv(out)

        if features:
            return prediction_map, feature_maps
        else:
            return prediction_map


class DownBlock2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size=4, stride=1, padding=1, norm=None):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        return out
