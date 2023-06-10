import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from .transformer import CrossAttention
from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear, ScaledLeakyReLU, EqualConv2d
import math


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class ResidualEncoder(Module):
    def __init__(self, opts=None):
        super(ResidualEncoder, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))

        self.conv_layer2 = Sequential(*[bottleneck_IR(32, 48, 2), bottleneck_IR(48, 48, 1), bottleneck_IR(48, 48, 1)])

        self.conv_layer3 = Sequential(*[bottleneck_IR(48, 64, 2), bottleneck_IR(64, 64, 1), bottleneck_IR(64, 64, 1)])

        self.condition_scale3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

        self.condition_shift3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = torch.nn.functional.interpolate(scale, size=(16, 16), mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = torch.nn.functional.interpolate(shift, size=(16, 16), mode='bilinear')
        conditions.append(shift.clone())
        return conditions


class GradualFeatureBlock(Module):
    def __init__(self, in_c, out_c, spatial_input, spatial_out):
        super(GradualFeatureBlock, self).__init__()
        self.out_c = out_c
        num_pools = abs(int(np.log2(spatial_out)) - int(np.log2(spatial_input)))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)

    def forward(self, x):
        x = self.convs(x)
        return x


class Decoder(Module):
    def __init__(self, in_c, out_c, deconv_nums, use_bias=False, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        model = []
        for i in range(deconv_nums):  # add upsampling layers
            model += [nn.ConvTranspose2d(in_c, out_c,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(out_c),
                      nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None, for_image=True, for_feature=True, for_ww=True):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.opts = opts
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc if opts is not None else 3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(self.opts.output_size, 2))
        # self.num_layers = (log_size - 2) * 2 + 1
        # self.style_count = opts.n_styles if opts is not None else 18
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 2
        self.middle_ind = 6
        self.styles.append(GradualStyleBlock(512, 512, 16))

        if not for_image:
            if for_feature:
                if opts.no_feature_attention:
                    print('###########################no_feature_attention ########################################')
                    self.content_layer = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.PReLU(num_parameters=512),
                        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                else:
                    if opts.no_res:
                        self.decoder = Decoder(in_c=512, out_c=512, deconv_nums=6)
                    self.content_layer = GradualFeatureBlock(512, 512, 64, 16)
                    self.feature_cross_attention = CrossAttention(d_model=512, nhead=4, dim_feedforward=1024)

            else:
                print('############################ not feature #######################')
            self.styles_others = nn.ModuleList()
            self.cross_attention = nn.ModuleList()
            if for_ww:
                for i in range(self.style_count - 1):
                    if i < self.coarse_ind:
                        style = GradualStyleBlock(512, 512, 16)
                    elif i < self.middle_ind:
                        style = GradualStyleBlock(512, 512, 32)
                    else:
                        style = GradualStyleBlock(512, 512, 64)
                    self.styles_others.append(style)
                if opts.no_w_attention:
                    print('###########################no_wwww_attention ########################################')

                else:
                    for i in range(self.style_count):
                        self.cross_attention.append(CrossAttention(d_model=512, nhead=4, dim_feedforward=1024))
            else:
                print('############################ not ww #######################')

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.image_encoder = for_image
        self.for_feature = for_feature
        self.for_ww =  for_ww

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _forward_image(self, x):
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        p2 = self._upsample_add(c3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer2(c1))
        out = self.styles[0](c3)
        return out

    def _forward_to_latent(self, x, edit_offset, no_res):
        feature_offset = None
        out_offset = None
        x = self.input_layer(x)
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        latent_base = self.styles[0](c3)
        if edit_offset is not None:
            if len(edit_offset.shape) == 3:
                edit_offset = edit_offset[:, 0, :]
            latent_base = latent_base + edit_offset
        if self.for_ww:
            latents.append(latent_base)
            for j in range(self.coarse_ind):
                latents.append(self.styles_others[j](c3))

            p2 = self._upsample_add(c3, self.latlayer1(c2))
            for j in range(self.coarse_ind, self.middle_ind):
                latents.append(self.styles_others[j](p2))

            p1 = self._upsample_add(p2, self.latlayer2(c1))
            for j in range(self.middle_ind, self.style_count - 1):
                latents.append(self.styles_others[j](p1))
            refine_offset = []
            latent_q = latent_base.unsqueeze(1).permute(1, 0, 2)
            if self.opts.no_w_attention:
                refine_offset = latents
            else:
                for k in range(self.style_count):
                    latent_v = latents[k].unsqueeze(1).permute(1, 0, 2)
                    refine_offset.append(self.cross_attention[k](latent_q, latent_v).permute(1, 0, 2))
            if self.for_feature:
                B, C, H, W = p1.shape
                if self.opts.no_feature_attention:
                    feature_offset = self.content_layer(p1)
                else:
                    p1 = p1.flatten(2).permute(2, 0, 1)
                    # if edit_offset is not None:
                    #     latent_base_f = latent_base.clone()
                    #     # latent_base_f = latent_base_f + 15 * edit_offset
                    #     latent_q = latent_base_f.unsqueeze(1).permute(1, 0, 2)
                    if no_res:
                        f_q = self.decoder(latent_base.unsqueeze(-1).unsqueeze(-1))  # B 64 64 512
                        f_q = f_q.flatten(2).permute(2, 0, 1)
                        feature_offset = self.feature_cross_attention(p1, f_q, no_res=no_res).permute(1, 0, 2)
                    else:
                        feature_offset = self.feature_cross_attention(p1, latent_q).permute(1, 0, 2)
                    feature_offset = feature_offset.transpose(-2, -1).contiguous().view(B, C, H, W)
                    if self.opts.dataset_type == "car_encode_inversion":
                        feature_offset = F.pad(feature_offset,  (0, 0, int((W - H) / 2), int((W - H) / 2)), "constant", 0)

                    feature_offset = self.content_layer(feature_offset)


            out_offset = torch.stack(refine_offset, dim=1)
        return out_offset, latent_base, feature_offset

    def forward(self, x, edit_offset=None, no_res=False):
        if self.image_encoder:
            return self._forward_image(x)
        else:
            return self._forward_to_latent(x, edit_offset, no_res)


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x


# unit test
if __name__ == '__main__':
    input_test = torch.rand(1, 3, 256, 256).cuda()
    model = GradualStyleEncoder(50, 'ir_se').cuda()
    output_test = model(input_test)
    print(output_test.shape)
