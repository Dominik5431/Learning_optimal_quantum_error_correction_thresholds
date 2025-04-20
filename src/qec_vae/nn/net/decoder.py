import copy

import torch
import torch.nn as nn
from torch.nn import MaxUnpool2d as MaxUnpool2d, ModuleList
import torch.nn.functional as F

""" 
This file contains different decoder architectures. 
"""


def d_conv_1(dis):
    return 8


def d_conv_2(dis):
    return 12


def d_ff(dis):
    return 20


class TransformerDecoder(nn.Module):
    """
    Decoder based upon Transformer Decoders.
    """
    def __init__(self, latent_dims, distance, channels):
        super(TransformerDecoder, self).__init__()

        num_layers = 2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # define structure
        self.linear = nn.Linear(latent_dims, (distance ** 2) * channels, device=device)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(distance ** 2, channels))

        self.decoders = ModuleList(
            [copy.deepcopy(DecoderLayer(channels, 1, 100, device=device)) for _ in range(num_layers)])

        # self.decoder = DecoderLayer(channels, 1, 100)
        self.lstm = nn.LSTM(channels, 10, batch_first=True, proj_size=channels, device=device)

    def forward(self, z):
        x = self.linear(z)
        x = self.unflatten(x)
        for d in self.decoders:
            x = d(x)
        x, _ = self.lstm(x)
        return x


class Decoder(nn.Module):
    """
    Decoder network containing two linear layers, two times two transpose convolutions and several batch norms.
    """
    def __init__(self, latent_dims, distance, channels, device: torch.device = torch.device('cpu')):
        super(Decoder, self).__init__()
        # self.linear2 = nn.Linear(latent_dims, 20 + 5)
        self.linear2 = nn.Linear(latent_dims, d_ff(distance))
        self.dropout = nn.Dropout(0.1)
        # self.bn4 = nn.BatchNorm1d(20 + 5)
        self.bn4 = nn.BatchNorm1d(d_ff(distance))
        self.linear1 = nn.Linear(d_ff(distance),
                                 (d_conv_2(distance) * int(0.25 * distance + 3/4) * int(0.25 * distance + 3/4)))
        self.bn3 = nn.BatchNorm1d((d_conv_2(distance) * int(0.25 * distance + 3/4) * int(0.25 * distance + 3/4)))
        self.unflatten = nn.Unflatten(1, (d_conv_2(distance), int(0.25 * distance + 3/4), int(0.25 * distance + 3/4)))

        # Don't use max_unpool anymore since the information flow in form of the pooling indices across the botleneck
        # distorts the result in the latent space
        # --> use F.interpolate in mode='nearest' instead
        # use nn.Upsampling

        # self.max_unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        # self.max_unpool2 = MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        # self.deconv3_2 = nn.ConvTranspose2d(20, 20, kernel_size=2, stride=1, padding=1, bias=True)
        # self.deconv3_1 = nn.ConvTranspose2d(20, 10, kernel_size=2, stride=1, padding=1, bias=True)

        self.bn2 = nn.BatchNorm2d(d_conv_2(distance))
        self.upsampling1 = nn.ConvTranspose2d(d_conv_2(distance), d_conv_2(distance), 4, 2, 1, bias=True, groups=d_conv_2(distance))  # nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv2_2 = nn.ConvTranspose2d(d_conv_2(distance), d_conv_2(distance), kernel_size=2, stride=1, padding=1,
                                            bias=True)
        self.deconv2_1 = nn.ConvTranspose2d(d_conv_2(distance), d_conv_1(distance), kernel_size=3, stride=1, padding=1,
                                            bias=True)
        self.bn1 = nn.BatchNorm2d(d_conv_1(distance))
        self.upsampling2 = nn.ConvTranspose2d(d_conv_1(distance), d_conv_1(distance), 4, 2, 1, bias=True, groups=d_conv_1(distance))  # nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv1_2 = nn.ConvTranspose2d(d_conv_1(distance), d_conv_1(distance), kernel_size=2, stride=1, padding=1,
                                            bias=True)
        self.deconv1_1 = nn.ConvTranspose2d(d_conv_1(distance), channels, kernel_size=3, stride=1, padding=1, bias=True)

        # self.linearl_2 = nn.Linear(5, 10)
        # self.linearl_1 = nn.Linear(10, 4 * distance)

    def forward(self, z, skip=None):
        x = F.relu(self.bn4(self.dropout(self.linear2(z))))

        x = F.relu(self.bn3(self.linear1(x)))

        x = self.unflatten(x)
        # x = self.max_unpool2(x, indices2, output_size=input_size2)
        # x = F.interpolate(x, size=(2 * x.size()[2] - 1, 2 * x.size()[3] - 1), mode='bilinear', align_corners=True)
        # x = F.relu(self.bn2(self.deconv3_1(F.relu(self.deconv3_2(x)))))

        # x = F.interpolate(x, size=(input_size2[2], input_size2[3]), mode='nearest')
        x = self.upsampling1(x)
        x = F.relu(self.bn1(self.deconv2_1(F.relu(self.deconv2_2(x)))))
        # x = F.interpolate(x, size=(input_size1[2], input_size1[3]), mode='nearest')
        x = self.upsampling2(x)
        x = F.tanh(self.deconv1_1(F.relu(self.deconv1_2(x))))
        return x


class DecoderSkip(nn.Module):
    """
    Decoder network containing two linear layers, two times two transpose convolutions and several batch norms.
    """

    def __init__(self, latent_dims, distance, channels, device: torch.device = torch.device('cpu')):
        super(DecoderSkip, self).__init__()
        # self.linear2 = nn.Linear(latent_dims, 20 + 5)
        self.linear2 = nn.Linear(latent_dims, d_ff(distance))
        self.dropout = nn.Dropout(0.1)
        # self.bn4 = nn.BatchNorm1d(20 + 5)
        self.bn4 = nn.BatchNorm1d(d_ff(distance))
        self.linear1 = nn.Linear(d_ff(distance),
                                 (d_conv_2(distance) * int(0.25 * distance + 3 / 4) * int(0.25 * distance + 3 / 4)))
        self.bn3 = nn.BatchNorm1d((d_conv_2(distance) * int(0.25 * distance + 3 / 4) * int(0.25 * distance + 3 / 4)))
        self.unflatten = nn.Unflatten(1,
                                      (d_conv_2(distance), int(0.25 * distance + 3 / 4), int(0.25 * distance + 3 / 4)))

        # Don't use max_unpool anymore since the information flow in form of the pooling indices across the botleneck
        # distorts the result in the latent space
        # --> use F.interpolate in mode='nearest' instead
        # use nn.Upsampling

        # self.max_unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        # self.max_unpool2 = MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        # self.deconv3_2 = nn.ConvTranspose2d(20, 20, kernel_size=2, stride=1, padding=1, bias=True)
        # self.deconv3_1 = nn.ConvTranspose2d(20, 10, kernel_size=2, stride=1, padding=1, bias=True)

        self.bn2 = nn.BatchNorm2d(d_conv_2(distance))
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv2_2 = nn.ConvTranspose2d(d_conv_2(distance), d_conv_2(distance), kernel_size=2, stride=1, padding=1,
                                            bias=True)
        self.deconv2_1 = nn.ConvTranspose2d(d_conv_2(distance), d_conv_1(distance), kernel_size=3, stride=1, padding=1,
                                            bias=True)
        self.bn1 = nn.BatchNorm2d(d_conv_1(distance))
        self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv1_2 = nn.ConvTranspose2d(d_conv_1(distance), d_conv_1(distance), kernel_size=2, stride=1, padding=1,
                                            bias=True)
        self.deconv1_1 = nn.ConvTranspose2d(d_conv_1(distance), channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.skip_norm = nn.BatchNorm2d(d_conv_1(distance))
        self.deconv_skip_2 = nn.ConvTranspose2d(d_conv_2(distance), d_conv_2(distance), kernel_size=2, stride=1, padding=1,
                                                bias=True)
        self.deconv_skip_1 = nn.ConvTranspose2d(d_conv_2(distance), d_conv_1(distance), kernel_size=3, stride=1, padding=1,
                                                bias=True)

        # self.linearl_2 = nn.Linear(5, 10)
        # self.linearl_1 = nn.Linear(10, 4 * distance)

    def forward(self, z, skip):
        assert not skip is None
        x = F.relu(self.bn4(self.dropout(self.linear2(z))))

        x = F.relu(self.bn3(self.linear1(x)))

        x = self.unflatten(x)
        # x = self.max_unpool2(x, indices2, output_size=input_size2)
        # x = F.interpolate(x, size=(2 * x.size()[2] - 1, 2 * x.size()[3] - 1), mode='bilinear', align_corners=True)
        # x = F.relu(self.bn2(self.deconv3_1(F.relu(self.deconv3_2(x)))))

        # x = F.interpolate(x, size=(input_size2[2], input_size2[3]), mode='nearest')
        x = self.upsampling1(x)
        x = F.relu(self.bn1(self.deconv2_1(F.relu(self.deconv2_2(x)))))
        # x = F.interpolate(x, size=(input_size1[2], input_size1[3]), mode='nearest')
        x = self.skip_norm(x + F.relu(self.deconv_skip_1(F.relu(self.deconv_skip_2(skip)))))

        x = self.upsampling2(x)
        x = F.tanh(self.deconv1_1(F.relu(self.deconv1_2(x))))
        return x


class DecoderSimple(nn.Module):
    """
    Decoder with linear layers and transpose convolutions with less parameters and therefore less complexity.
    """
    def __init__(self, latent_dims: int, distance: int, channels: int):
        super(DecoderSimple, self).__init__()
        self.linear2 = nn.Linear(latent_dims, 20)
        self.dropout = nn.Dropout(0.3)
        self.linear_log_2 = nn.Linear(20, 20)
        self.linear_log_1 = nn.Linear(20, 4 * distance)
        # self.linear1 = nn.Linear(100, 10 * int(0.5 * (distance + 2)) * int(0.5 * (distance + 3)))
        # self.unflatten = nn.Unflatten(1, (10, int(0.5 * (distance + 2)), int(0.5 * (distance + 3))))
        self.linear1 = nn.Linear(20, 10 * int(0.5 * (distance + 3)) * int(0.5 * (distance + 3)))
        self.unflatten = nn.Unflatten(1, (10, int(0.5 * (distance + 3)), int(0.5 * (distance + 3))))
        # self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.max_unpool = MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv = nn.ConvTranspose2d(10, channels, kernel_size=2, stride=1, padding=1, bias=True)

    def forward(self, z, skip=None):
        x = F.relu(self.linear2(z))
        x = self.dropout(x)
        s = F.relu(self.linear1(x))
        s = self.unflatten(s)
        # s = self.max_unpool(s, indices1, output_size=input_size1)
        s = F.tanh(self.deconv(s))  # changed from tanh to sigmoid on 29.04., undone 08.05. --> rather MSE than BCE
        l = F.relu(self.linear_log_2(x)) #+ inf_trans
        l = F.tanh(self.linear_log_1(l))
        return s, l


class DecoderIsing(nn.Module):
    """
    Decoder that only contains transpose convolutions.
    Attention: Does not work with any distance due to stride=2!
    """
    def __init__(self, latent_dims, distance, channels):
        super(DecoderIsing, self).__init__()
        self.linear = nn.Linear(latent_dims, 128 * int((distance - 1) / 8 + 1) * int((distance - 1) / 8 + 1))
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(128, int((distance - 1) / 8 + 1), int((distance - 1) / 8 + 1)))
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, z, skip=None):
        x = F.relu(self.linear(z))
        x = self.unflatten(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.tanh(self.conv3(x))
        return x
