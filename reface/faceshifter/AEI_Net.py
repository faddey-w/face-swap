import torch
import torch.nn as nn
import torch.nn.functional as F
from reface import utils, face_recognizer
from reface.config import Config
from .AADLayer import AAD_ResBlk


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv4x4(in_c, out_c, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, image, skip):
        x = self.deconv(image)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class MLAttrEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super(MLAttrEncoder, self).__init__()
        first_dim = cfg.GENERATOR.DIMS[0]
        out_dims = cfg.GENERATOR.DIMS[1:]
        self.conv0 = conv4x4(cfg.INPUT.N_CHANNELS, first_dim)
        conv_stack = []
        deconv_stack = []
        for i, out_dim in enumerate(out_dims, 1):
            in_dim = cfg.GENERATOR.DIMS[i - 1]
            is_not_innermost = i != len(out_dims)
            conv_stack.append(conv4x4(in_dim, out_dim))
            deconv_stack.append(deconv4x4((1 + is_not_innermost) * out_dim, in_dim))
        self.conv_stack = nn.ModuleList(conv_stack)
        self.deconv_stack = nn.ModuleList(deconv_stack[::-1])

        self.apply(weight_init)

    def forward(self, img_target):
        feats = [self.conv0(img_target)]
        for conv in self.conv_stack:
            feats.append(conv(feats[-1]))
        z_attrs = [feats.pop(-1)]
        for deconv in self.deconv_stack:
            z_attrs.append(deconv(z_attrs[-1], feats.pop(-1)))
        z_attrs.append(
            F.interpolate(
                z_attrs[-1], scale_factor=2, mode="bilinear", align_corners=True
            )
        )
        return z_attrs


class AADGenerator(nn.Module):
    def __init__(self, cfg: Config):
        super(AADGenerator, self).__init__()
        c_id = face_recognizer.FaceRecognizer.embedding_dimension
        adaptive_norm = cfg.GENERATOR.AAD_USE_ADAPTIVE_NORM

        up1_size = cfg.INPUT.IMAGE_SIZE // (2 ** len(cfg.GENERATOR.DIMS))
        self.up1 = nn.ConvTranspose2d(
            c_id, cfg.GENERATOR.DIMS[-1], kernel_size=up1_size, stride=1, padding=0
        )
        dims = cfg.GENERATOR.DIMS[::-1]
        self.aad_blocks = nn.ModuleList(
            [AAD_ResBlk(dims[0], dims[0], dims[0], c_id, adaptive_norm=adaptive_norm)]
        )
        last_cout = dims[0]
        for i in range(len(dims) - 1):
            cin, cout, c_attr = last_cout, dims[i], 2 * dims[i + 1]
            self.aad_blocks += [
                AAD_ResBlk(cin, cout, c_attr, c_id, adaptive_norm=adaptive_norm)
            ]
            last_cout = cout
        self.last_aad_block = AAD_ResBlk(
            last_cout, 3, 2 * dims[-1], c_id, adaptive_norm=adaptive_norm
        )
        self.apply(weight_init)

    def forward(self, z_attrs, z_id):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        for add_block, z_attr in zip(self.aad_blocks, z_attrs):
            m = F.interpolate(
                add_block(m, z_attr, z_id),
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        y = self.last_aad_block(m, z_attrs[-1], z_id)
        return torch.tanh(y)


class AEI_Net(nn.Module):
    def __init__(self, cfg):
        super(AEI_Net, self).__init__()
        self.encoder = MLAttrEncoder(cfg)
        self.generator = AADGenerator(cfg)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        return self.encoder(X)
