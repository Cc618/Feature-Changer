import torch as T
from torch import nn
import torch.nn.functional as F
from params import z_size, img_size, img_depth

class Debug(nn.Module):
    '''
    Debug shape layer
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(f'Debug : {x.shape}')

        return x


class Net1(nn.Module):
    '''
    Simple fully convolutional model
    - decoder : Stride can be 2 to reduce the size
    - encoder : Upsample is used to increase the size
    '''
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
                nn.Conv2d(img_depth, 32, 4, 2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(.2),

                nn.Conv2d(32, 16, 4),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(.2),

                nn.Conv2d(16, 4, 4),
                nn.BatchNorm2d(4),
                nn.LeakyReLU(.2),
            )

        self.decode = nn.Sequential(
                nn.Upsample(scale_factor=2),

                nn.Conv2d(4, 32, 2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(.2),
                nn.Upsample(scale_factor=2),

                nn.Conv2d(32, 16, 2),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(.2),

                nn.Conv2d(16, img_depth, 2),
            )

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y

class Net2(nn.Module):
    '''
    https://github.com/ku2482/vae.pytorch/blob/master/models/simple_vae.py
    '''
    def __init__(self):
        super().__init__()

        nc = 3
        nef = 32
        ndf = nef
        nz = z_size

        self.ndf = ndf
        self.out_size = img_size // 16

        # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef),

            nn.Conv2d(nef, nef*2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*2),

            nn.Conv2d(nef*2, nef*4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*4),

            nn.Conv2d(nef*4, nef*8, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*8),
        )

        self.encoder_dense = nn.Linear(nef*8*self.out_size*self.out_size, nz)

        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf*8*self.out_size*self.out_size),
            nn.ReLU(True)
        )

        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf*8, ndf*4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf*4, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf*4, ndf*2, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf*2, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf*2, ndf, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1),
        )

    def encode(self, x):
        z = self.encoder_conv(x)
        z = z.view(z.size(0), -1)
        z = self.encoder_dense(z)

        return z

    def decode(self, z):
        y = self.decoder_dense(z)
        y = y.view(-1, self.ndf*8, self.out_size, self.out_size)
        y = self.decoder_conv(y)

        return y

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y

class Net3(nn.Module):
    '''
    Lighter Net2
    '''
    def __init__(self):
        super().__init__()

        nc = 3
        nef = 32
        ndf = nef
        nz = z_size

        self.ndf = ndf
        self.out_size = img_size // 8

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef),

            nn.Conv2d(nef, nef*2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*2),

            nn.Conv2d(nef*2, nef*4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(nef*4),
        )

        self.encoder_dense = nn.Linear(nef*4*self.out_size*self.out_size, nz)

        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf*4*self.out_size*self.out_size),
            nn.ReLU(True)
        )

        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf*4, ndf*2, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf*2, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf*2, ndf, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ndf, 1.e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1),
        )

#     def forward(self, x):
#         # Encode
#         z = self.encoder_conv(x)
#         # Shape : 128x4x4
#         z = z.view(z.size(0), -1)
#         z = self.encoder_dense(z)
#         # Shape : 100

#         # Decode
#         y = self.decoder_dense(z)
#         # Shape : 2048
#         y = y.view(-1, self.ndf*4, self.out_size, self.out_size)
#         y = self.decoder_conv(y)

#         return y

    def encode(self, x):
        z = self.encoder_conv(x)
        # Shape : 128x4x4
        z = z.view(z.size(0), -1)
        z = self.encoder_dense(z)

        return z

    def decode(self, z):
        y = self.decoder_dense(z)
        # Shape : 2048
        y = y.view(-1, self.ndf*4, self.out_size, self.out_size)
        y = self.decoder_conv(y)

        return y

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y


# --- Progressive Growing Autoencoder ---
def FromRGB(chan):
    return nn.Conv2d(3, chan, 1)


def ToRGB(chan):
    return nn.Conv2d(chan, 3, 1)


def EConv(in_chan, out_chan):
    '''
    Encoder convolution block (no downsampling performed)
    '''
    return nn.Sequential(
            # Conv 1
            nn.Conv2d(in_chan, out_chan, 3, padding=1),
            nn.LeakyReLU(.2),
            nn.BatchNorm2d(out_chan),

            # Conv 2
            nn.Conv2d(out_chan, out_chan, 3, padding=1),
            nn.LeakyReLU(.2),
            nn.BatchNorm2d(out_chan),
        )


def DConv(in_chan, out_chan):
    '''
    Decoder convolution block (no upsampling performed)
    '''
    return nn.Sequential(
            # Conv 1
            nn.Conv2d(in_chan, out_chan, 3, padding=1),
            nn.LeakyReLU(.2),
            nn.BatchNorm2d(out_chan),

            # Conv 2
            nn.Conv2d(out_chan, out_chan, 3, padding=1),
            nn.LeakyReLU(.2),
            nn.BatchNorm2d(out_chan),
        )


class EBlock(nn.Module):
    '''
    Encoder block
    '''
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.from_rgb = FromRGB(in_chan)
        self.conv = EConv(in_chan, out_chan)

    def forward(self, x, rgb_input=False):
        y = x

        if rgb_input:
            y = self.from_rgb(y)

        y = self.conv(y)
        y = F.max_pool2d(y, 2)

        return y


class DBlock(nn.Module):
    '''
    Decoder block
    '''
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = DConv(in_chan, out_chan)
        self.to_rgb = ToRGB(out_chan)

    def forward(self, x, rgb_output=False):
        y = x

        y = F.interpolate(y, scale_factor=2, mode='bilinear',
                align_corners=False)
        y = self.conv(y)

        if rgb_output:
            y = self.to_rgb(y)

        return y


class PGAE(nn.Module):
    '''
    Progressive Auto Encoder
    '''
    def __init__(self, steps, chan):
        super().__init__()

        self.steps = steps

        self.encoders = nn.ModuleList([
                EBlock(chan * 2 ** step, chan * 2 ** (step + 1))
                for step in range(steps)
            ])

        self.decoders = nn.ModuleList([
                DBlock(chan * 2 ** step, chan * 2 ** (step - 1))
                for step in range(steps, 0, -1)
            ])

        # How much we merge the last encoding / first decoding layer
        # alpha in the paper
        self.merge_ratio = 1

        # Current step
        # decoders[:step] will be applied
        self.step = 1

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y

    def encode(self, x):
        z = x

        # Merge if at least 2 layers
        if self.step > 1 and self.merge_ratio < 1:
            z_prev = F.avg_pool2d(z, 2)
            z_prev = self.encoders[self.steps - self.step + 1].from_rgb(z_prev)

            z = self.encoders[self.steps - self.step](z, rgb_input=True)

            z = self.merge_ratio * z + (1 - self.merge_ratio) * z_prev
        else:
            z = self.encoders[self.steps - self.step](z, rgb_input=True)

        for i in range(self.steps - self.step + 1, self.steps):
            z = self.encoders[i](z)

        return z

    def decode(self, z):
        y = z

        for i in range(self.step - 1):
            y = self.decoders[i](y)

        # Merge if at least 2 layers
        if self.step != 1:
            y_rgb_prev = self.decoders[self.step - 2].to_rgb(y)
            y_rgb_prev = F.interpolate(y_rgb_prev, scale_factor=2)

            y_rgb = self.decoders[self.step - 1](y, rgb_output=True)

            y = self.merge_ratio * y_rgb + (1 - self.merge_ratio) * y_rgb_prev
        else:
            y = self.decoders[self.step - 1](y, rgb_output=True)

        return y
