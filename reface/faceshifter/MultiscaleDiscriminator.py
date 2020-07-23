import torch.nn as nn
import numpy as np
from reface import utils
from reface.config import Config


class NLayerDiscriminator(nn.Module):
    def __init__(self, cfg: Config):
        super(NLayerDiscriminator, self).__init__()

        kw = cfg.DISCRIMINATOR.CONV_SIZE
        padw = int(np.ceil((kw - 1.0) / 2))
        lrelu_slope = cfg.DISCRIMINATOR.LEAKY_RELU_SLOPE
        norm_layer = utils.get_norm_cls(cfg.DISCRIMINATOR.NORM_LAYER)

        self.layers = nn.ModuleList()
        self.layers += [
            nn.Sequential(
                nn.Conv2d(
                    cfg.INPUT.N_CHANNELS,
                    cfg.DISCRIMINATOR.BASE_DIM,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                ),
                nn.LeakyReLU(lrelu_slope),
            )
        ]

        nf = cfg.DISCRIMINATOR.BASE_DIM
        for n in range(1, cfg.DISCRIMINATOR.N_LAYERS):
            nf_prev = nf
            nf = min(nf * 2, cfg.DISCRIMINATOR.MAX_DIM)
            self.layers += [
                nn.Sequential(
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(lrelu_slope),
                )
            ]

        nf_prev = nf
        nf = min(nf * 2, cfg.DISCRIMINATOR.MAX_DIM)
        self.layers += [
            nn.Sequential(
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(lrelu_slope),
            )
        ]

        self.layers += [
            nn.Sequential(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))
        ]

        if cfg.DISCRIMINATOR.USE_SIGMOID:
            self.layers += [nn.Sigmoid()]

    def forward(self, image, get_intermediate_features=False):
        res = [image]
        # noinspection PyTypeChecker
        for layer in self.layers:
            res.append(layer(res[-1]))
        if get_intermediate_features:
            return res[1:]
        else:
            return res[-1]


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, cfg: Config):
        super(MultiscaleDiscriminator, self).__init__()
        self.scales = nn.ModuleList(
            [NLayerDiscriminator(cfg) for _ in range(cfg.DISCRIMINATOR.N_SCALES)]
        )

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(self, image):
        result = []
        input_downsampled = image
        # noinspection PyTypeChecker
        for i, scale_discr in enumerate(self.scales):
            if i != 0:
                input_downsampled = self.downsample(input_downsampled)
            result.append(scale_discr(input_downsampled))
        return result
