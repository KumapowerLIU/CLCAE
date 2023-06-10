"""
This file defines the core research contribution
"""
import matplotlib
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append('/apdcephfs/share_1290939/kumamzqliu/code/pixel2style2pixel/models')
matplotlib.use('Agg')
import torch
from models.base_network import BaseNetwork
from models.encoders import psp_encoders
from configs.paths_config import model_paths
from models.stylegan2.model import Generator
import math


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class AFPSP(BaseNetwork):

    def __init__(self, opts):
        super(AFPSP, self).__init__()
        self.for_feature = not opts.not_for_feature
        self.for_ww = not opts.not_for_ww
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.log_size = int(math.log(self.opts.output_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_styles = self.log_size * 2 - 2
        if opts.idx_k is None:
            self.idx_k = 5
        else:
            self.idx_k = opts.idx_k

        # Load weights if needed n_in: int, n_hidden: int, n_out: int

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts, for_image=False,
                                                       for_feature=self.for_feature)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path_af is not None:
            print('Loading AFpSp from checkpoint: {}'.format(self.opts.checkpoint_path_af))
            ckpt = torch.load(self.opts.checkpoint_path_af, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            ckpt = torch.load(self.opts.stylegan_weights)
            self.__load_latent_avg(ckpt)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=self.n_styles)
            # if self.for_feature:
            #     for name, parameter in self.encoder.named_parameters():
            #         if 'feature_cross_attention' in name or 'content_layer' in name:
            #             parameter.requires_grad = True
            #         else:
            #             parameter.requires_grad = False
            # if self.for_ww:
            #     for name, parameter in self.encoder.named_parameters():
            #         if 'styles_others' in name or 'cross_attention' in name:
            #             parameter.requires_grad = True
            #         else:
            #             parameter.requires_grad = False
        else:
            print('Loading encoders weights from pretrained_contrastive!')

            encoder_ckpt = torch.load(model_paths[self.opts.contrastive_model_image])
            # encoder_ckpt = torch.load(model_paths['ir_se50'])

            # if input to encoder is not an RGB image, do not load the input layer weights

            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=self.n_styles)

    def forward(self, x, x_avg, resize=True, randomize_noise=True, return_latents=False, return_features=False,
                interation=0, edit_offset=None):
        images_ww = None
        images_w = None
        latent_offset, base_offset, feature_offset = self.encoder(x - x_avg, edit_offset=edit_offset,
                                                                  no_res=self.opts.no_res)
        latent_base = base_offset + self.latent_avg.repeat(base_offset.shape[0], 1)
        latent_base = latent_base.unsqueeze(1).repeat(1, self.n_styles, 1)

        latent_refine = latent_offset.squeeze(2) + latent_base
        # normalize with respect to the center of an average face
        features_offsets = [None] * self.idx_k + [feature_offset] + [None] * (self.num_layers - self.idx_k)
        if edit_offset is not None:
            latent_offset_original, base_offset_original, feature_offset_original = self.encoder(x - x_avg,
                                                                                                 edit_offset=None,
                                                                                                 no_res=self.opts.no_res)
            latent_base_original = base_offset_original + self.latent_avg.repeat(base_offset_original.shape[0], 1)
            latent_base_original = latent_base_original.unsqueeze(1).repeat(1, self.n_styles, 1)

            latent_refine_original = latent_offset_original.squeeze(2) + latent_base_original
            _, features_refine_original = self.decoder([latent_refine_original],
                                                       input_is_latent=True,
                                                       return_features=True,
                                                       randomize_noise=randomize_noise,
                                                       return_latents=return_latents)
            _, features_refine = self.decoder([latent_refine],
                                              input_is_latent=True,
                                              return_features=True,
                                              randomize_noise=randomize_noise,
                                              return_latents=return_latents)
            features_offsets = [None] * self.idx_k + [
                feature_offset + features_refine[self.idx_k] - features_refine_original[self.idx_k]] + [None] * (
                                           self.num_layers - self.idx_k)

        if self.for_feature:
            images, features_refine = self.decoder([latent_refine],
                                                   input_is_latent=True,
                                                   return_features=True,
                                                   randomize_noise=randomize_noise,
                                                   return_latents=return_latents,
                                                   features_in=features_offsets,
                                                   feature_scale=min(1.0, interation))

            images_ww, _ = self.decoder([latent_refine],
                                        input_is_latent=True,
                                        return_features=True,
                                        randomize_noise=randomize_noise,
                                        return_latents=return_latents)

            images_w, _ = self.decoder([latent_base],
                                       input_is_latent=True,
                                       return_features=True,
                                       randomize_noise=randomize_noise,
                                       return_latents=return_latents)
        elif self.for_ww:
            images, features_refine = self.decoder([latent_refine],
                                                   input_is_latent=True,
                                                   return_features=True,
                                                   randomize_noise=randomize_noise,
                                                   return_latents=return_latents)
            images_w, _ = self.decoder([latent_base],
                                       input_is_latent=True,
                                       return_features=True,
                                       randomize_noise=randomize_noise,
                                       return_latents=return_latents)

        else:
            images, features_refine = self.decoder([latent_base],
                                                   input_is_latent=True,
                                                   return_features=True,
                                                   randomize_noise=randomize_noise,
                                                   return_latents=return_latents)

        feature_refine = features_refine[self.idx_k].detach()


        if resize:
            images = self.face_pool(images)
            if images_w is not None:
                images_w = self.face_pool(images_w)
            if images_ww is not None:
                images_ww = self.face_pool(images_ww)
        if self.opts.dataset_type == "car_encode_inversion":
            images_ww = images_ww[:, :, 32:224, :]
            images_w = images_w[:, :, 32:224, :]
            images = images[:, :, 32:224, :]
        if return_latents:
            return images, latent_refine
        elif return_features:
            return images, latent_refine, latent_base, feature_offset, feature_refine, images_ww, images_w
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None


# unit_test
if __name__ == '__main__':
    class opt():
        encoder_type = 'GradualStyleEncoder'
        input_nc = 3
        n_styles = 18


    Opt = opt()
    print(Opt.encoder_type)
    model = AFPSP(opts=Opt).cuda()
    image = torch.rand(1, 3, 256, 256).cuda()
    image_avg = torch.rand(1, 3, 256, 256).cuda()
    out = model(image, image_avg)
    print(out.shape)
