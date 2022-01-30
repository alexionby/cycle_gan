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
    def __init__(self, f=64, blocks=9):
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
        self.down_x2 = nn.Upsample(size=128)
        
    def forward(self, x):
        
        out = self.conv(x)
        out_down_x2 = self.down_x2(out)

        return out, out_down_x2


class Discriminator(nn.Module):  
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc,ndf,4,2,1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf,ndf*2,4,2,1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4,ndf*8,4,1,1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8,1,4,1,1)
            # 128 -> 1 x 14 x 14 / 256 -> 1 x 30 x 30
        )

    def forward(self, input):
        return self.main(input)