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
