import torch
import math
import torch.nn.functional as F
from models.stylegan2.op import fused_leaky_relu
from models.encoders.transformer import TransformerEncoderLayer
import torch.nn as nn
from .encoders.projection_head import Projection
from .base_network import BaseNetwork


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class EncoderMlp(nn.Module):
    def __init__(self, dim, n_mlp, lr_mlp=0.01):
        super().__init__()
        layers = []

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    dim, dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.latent_encoder = nn.Sequential(*layers)

    def forward(self, x, x_avg):
        out = self.latent_encoder(x - x_avg)
        return out


class LatentEncoder(BaseNetwork):
    def __init__(self, opts):
        super(LatentEncoder, self).__init__()
        self.set_opts(opts)
        num_latent = self.opts.num_latent
        dim = self.opts.latent_embedding_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_latent, dim))
        self.coarse = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024)
        self.medium = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024)
        self.fine = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024)
        self.latent_projection = Projection(n_in=opts.latent_embedding_dim, n_hidden=opts.latent_embedding_dim,
                                            n_out=opts.latent_embedding_dim)

    def load_weights(self, model_path=None):
        if self.opts.checkpoint_path_latent is not None or model_path is not None:
            if model_path is not None:
                pretrained_model = model_path
            else:
                pretrained_model = self.opts.checkpoint_path_latent
            print('Loading latentencoder from checkpoint: {}'.format(pretrained_model))
            ckpt = torch.load(pretrained_model, map_location='cpu')
            self.coarse.load_state_dict(get_keys(ckpt, 'coarse'), strict=True)
            self.medium.load_state_dict(get_keys(ckpt, 'medium'), strict=True)
            self.fine.load_state_dict(get_keys(ckpt, 'fine'), strict=True)
            self.latent_projection.load_state_dict(get_keys(ckpt, 'latent_projection'), strict=True)

            self.pos_embedding = nn.Parameter(get_keys(ckpt, 'pos_embedding')[''])
        else:
            print('No former trained model')
            pass

    def forward(self, x, x_avg, return_all=False):
        latent_input = x - x_avg
        latent_input = latent_input.permute(1, 0, 2)  # N B C
        latent_coarse = self.coarse(latent_input, pos=self.pos_embedding)
        latent_medium = self.medium(latent_coarse, pos=self.pos_embedding)
        latent_fine = self.fine(latent_medium, pos=self.pos_embedding)
        latent_out = latent_fine.permute(1, 0, 2)
        latent_out = latent_out.squeeze(1)
        latent_feature = self.latent_projection(latent_out)
        if self.opts.use_norm:
            latent_feature = F.normalize(latent_feature, dim=-1)
        if return_all:
            return latent_out, latent_coarse.permute(1, 0, 2), latent_medium.permute(1, 0, 2)
        else:
            return latent_feature

    def set_opts(self, opts):
        self.opts = opts


# unit test
if __name__ == '__main__':
    model = LatentEncoder().cuda()
    input_test = torch.rand(1, 1, 512).cuda()
    latent_avg = torch.rand(1, 1, 512).cuda()
    out = model(input_test, latent_avg)
    print(out.shape)
