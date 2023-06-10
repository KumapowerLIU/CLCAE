"""
This file defines the core research contribution
"""
import matplotlib

matplotlib.use('Agg')
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .base_network import BaseNetwork
from models.encoders import psp_encoders
from configs.paths_config import model_paths
from .encoders.projection_head import ProjectionHead, Projection


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class ImageEncoder(BaseNetwork):

    def __init__(self, opts):
        super(ImageEncoder, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed n_in: int, n_hidden: int, n_out: int
        self.image_projection = Projection(n_in=opts.image_embedding_dim, n_hidden=opts.image_embedding_dim,
                                           n_out=opts.image_embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self, model_path=None):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        if self.opts.checkpoint_path_image is not None or model_path is not None:
            if model_path is not None:
                pretrained_model = model_path
            else:
                pretrained_model = self.opts.checkpoint_path_image
            print('Loading latentencoder from checkpoint: {}'.format(pretrained_model))
            ckpt = torch.load(pretrained_model, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.image_projection.load_state_dict(get_keys(ckpt, 'image_projection'), strict=True)
            self.logit_scale = nn.Parameter(get_keys(ckpt, 'logit_scale')[''])
        elif self.opts.load_pretrain_image_encoder:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading W_avg from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)
        else:
            print('No former trained model')

    def forward(self, x, x_avg):
        out = self.encoder(x - x_avg)
        image_feature = self.image_projection(out)
        if self.opts.use_norm:
            image_feature = F.normalize(image_feature, dim=-1)
        return image_feature, self.logit_scale.exp()

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
