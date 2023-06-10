import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from models.image_encoder import ImageEncoder
from models.latent_encoder import LatentEncoder
from .contrastive_loss import ClipLoss


class ContrastiveID(nn.Module):
    def __init__(self, opts):
        super(ContrastiveID, self).__init__()
        self.net_image = ImageEncoder(opts)
        self.net_latent = LatentEncoder(opts)
        self.net_image.load_weights(model_paths[opts.contrastive_model_image])

        print('Loading decoder weights from pretrained contrastive_image!')
        self.net_latent.load_weights(model_paths[opts.contrastive_model_latent])
        print('Loading decoder weights from pretrained contrastive_latent!')
        self.net_image.eval()
        self.net_latent.eval()
        for param in self.net_image.parameters():
            param.requires_grad = False
        for latent_encoder in self.net_latent.parameters():
            latent_encoder.requires_grad = False
        self.contrastive_loss = ClipLoss()

    def forward(self, image, image_avg, latent, latent_avg):
        '''

        Args:
            image: True
            latent: Fake

        Returns:

        '''
        B = image.shape[0]
        latent_avg = latent_avg.repeat(B, 1)
        latent_avg = latent_avg.unsqueeze(1)
        image_embedding, t = self.net_image.forward(image, image_avg)
        latent_embedding = self.net_latent.forward(latent, latent_avg)
        loss = self.contrastive_loss(image_embedding, latent_embedding, t)
        return loss
