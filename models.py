import torch
import torch.nn as nn
import torch.nn.functional as F

nc = 3
ndf = 64
norm_layer = nn.BatchNorm2d


class ConvNormRelu(nn.Module):
    def __init__(self, inf, onf, ks=3, stride=1, padding=1):
        super(ConvNormRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inf, onf,
                                            kernel_size=ks,
                                            stride=stride,
                                            padding=padding),
                                  norm_layer(onf),
                                  nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class ConvTpNormRelu(nn.Module):
    def __init__(self, inf, onf, ks=3, stride=1, padding=1, output_padding=1):
        super(ConvTpNormRelu, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(inf, onf, ks,
                                                     stride=stride,
                                                     padding=padding,
                                                     output_padding=output_padding),
                                  norm_layer(onf),
                                  nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class UpConvNormRelu(nn.Module):
    def __init__(self, inf, onf, ks=3, scale=2, padding=1, output_padding=1):
        super(UpConvNormRelu, self).__init__()
        self.conv = nn.Sequential(nn.Upsample(scale_factor=scale, mode='nearest'),
                                  nn.Conv2d(inf, onf, ks, stride=1,
                                            padding=padding),
                                  norm_layer(onf),
                                  nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class DepthBlock(nn.Module):
    def __init__(self, nf, kernel, reduction):
        super(DepthBlock, self).__init__()
        pad = int((kernel - 1) // 2)
        rnf = int(nf // reduction)

        block = [nn.Conv2d(nf, rnf, 1, 1, padding=0),
                norm_layer(rnf),
                nn.ReLU(True),
                nn.Conv2d(rnf, rnf, kernel, 1, padding=pad, groups=rnf),
                norm_layer(rnf),
                nn.ReLU(True),
                nn.Conv2d(rnf, nf, 1, 1, padding=0)]

        self.block = nn.Sequential(*block)


    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel, reduction):
        super(ResidualBlock, self).__init__()
        pad = int((kernel - 1) // 2)
        rnf = int(nf // reduction)

        block = [nn.Conv2d(nf, rnf, kernel, 1, padding=pad),
                norm_layer(rnf),
                nn.ReLU(True),
                nn.Conv2d(rnf, nf, kernel, 1, padding=pad)]

        self.block = nn.Sequential(*block)


    def forward(self, x):
        return self.block(x)


class InceptionBlock(nn.Module):
    def __init__(self, nf,
                 dw_kernels=[1, 3, 5],
                 resnet_kernels=[1, 3, 5],
                 reduction=4):
        super(InceptionBlock, self).__init__()

        dw_blocks = []
        for kernel in dw_kernels:
            dw_blocks += [DepthBlock(nf, kernel, reduction)]

        res_blocks = []
        for kernel in resnet_kernels:
            res_blocks += [ResidualBlock(nf, kernel, reduction)]

        self.res_blocks = nn.Sequential(*res_blocks)
        self.dw_blocks = nn.Sequential(*dw_blocks)
        self.last_conv = nn.Conv2d(nf, nf, 1)
        self.norm = norm_layer(nf)

    def forward(self, x):
        tmp = sum([op(x) for op in self.dw_blocks]) + sum([op(x) for op in self.res_blocks])
        tmp = self.last_conv(tmp)
        tmp = x + self.norm(tmp)
        return tmp


class InceptionGenerator(nn.Module):
    def __init__(self, nf=64,
                 blocks=9,
                 num_down=2,
                 image_size=256,
                 reduction=6):
        super(InceptionGenerator, self).__init__()

        downsample = []

        # downsample.append(nn.ReflectionPad2d(3))
        downsample.append(ConvNormRelu(3, nf, 7, 1, 3))

        for i in range(0, num_down):
            imult = 2**i
            omult = 2**(i+1)
            downsample.append(ConvNormRelu(imult * nf, omult * nf, 3, 2, 1))

        features = []
        for i in range(blocks):
            fnf = (2**num_down) * nf
            features += [InceptionBlock(fnf, reduction=reduction)]

        upsample = []
        for i in range(0, num_down)[::-1]:
            omult = 2**i
            imult = 2**(i+1)
            upsample.append(UpConvNormRelu(imult * nf, omult * nf, 3,
                                           scale=2,
                                           padding=1,
                                           output_padding=1))

        # upsample.append(nn.ReflectionPad2d(3))
        upsample.append(nn.Conv2d(nf, 3, 7, 1, 3))
        upsample.append(nn.Tanh())

        self.down = nn.Sequential(*downsample)
        self.features = nn.Sequential(*features)
        self.up = nn.Sequential(*upsample)

        self.down_x2 = nn.Upsample(size=int(image_size/2))
        self.down_x4 = nn.Upsample(size=int(image_size/4))

    def forward(self, x):
        out = self.down(x)
        out = self.features(out)
        out = self.up(out)
        out_down_x2 = self.down_x2(out)
        out_down_x4 = self.down_x4(out)
        return out, out_down_x2, out_down_x4


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
            DownBlock2d(nc, ndf, 4, 2, 1, False)
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


if __name__ == "__main__":

    model = InceptionGenerator(64, reduction=8)
    print(model)

    rand = torch.FloatTensor(1, 3, 256, 256)
    out = model(rand)

    torch.onnx.export(model, rand, "model.onnx", opset_version=9)