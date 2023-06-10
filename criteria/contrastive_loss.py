import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        latent_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_latent_features = hvd.allgather(latent_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_latent_features = hvd.allgather(latent_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_latent_features = list(all_latent_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_latent_features[rank] = latent_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_latent_features = torch.cat(gathered_latent_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_latent_features = torch.cat(torch.distributed.nn.all_gather(latent_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_latent_features = [torch.zeros_like(latent_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_latent_features, latent_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_latent_features[rank] = latent_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_latent_features = torch.cat(gathered_latent_features, dim=0)

    return all_image_features, all_latent_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, latent_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_latent_features = gather_features(
                image_features, latent_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_latent_features.T
                logits_per_latent = logit_scale * latent_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_latent_features.T
                logits_per_latent = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ latent_features.T
            logits_per_latent = logit_scale * latent_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_latent, labels)
                     ) / 2
        return total_loss


class SimpleContrastiveLoss(nn.Module):
    def __init__(self):
        super(SimpleContrastiveLoss, self).__int__()

    def forward(self, text_embeddings, image_embeddings, temperature):
        logits = (text_embeddings @ image_embeddings.T) / temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


import torch.nn as nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from models.image_encoder import ImageEncoder
from models.latent_encoder import LatentEncoder


# class ContrastiveID(nn.Module):
#     def __init__(self, opts):
#         self.net_image = ImageEncoder(opts)
#         self.net_latent = LatentEncoder(opts)
#         self.net_image.load_weights(model_paths['contrastive_image'])
#
#         print('Loading decoder weights from pretrained contrastive_image!')
#         self.net_latent.load_weights(model_paths['contrastive_latent'])
#         print('Loading decoder weights from pretrained contrastive_latent!')
#         self.net_image.eval()
#         self.net_latent.eval()
#         for param in self.net_image.parameters():
#             param.requires_grad = False
#         for latent_encoder in self.net_latent.parameters():
#             latent_encoder.requires_grad = False
#         self.contrastive_loss = ClipLoss()
#
#     def forward(self, image, image_avg, latent, latent_avg):
#         '''
#
#         Args:
#             image: True
#             latent: Fake
#
#         Returns:
#
#         '''
#         B = image.shape[0]
#         latent_avg = latent_avg.repeat(B, 1)
#         image_embedding, t = self.net_image.forward(image, image_avg)
#         latent_embedding = self.net_latent.forward(latent, latent_avg)
#         loss = self.contrastive_loss(image_embedding, latent_embedding, t)
#         return loss

# unit_test
if __name__ == '__main__':
    loss = ClipLoss().cuda()
    image_test = torch.rand(1,  256).cuda()
    latent_test = torch.rand(1, 256).cuda()
    t = 1/0.07
    out = loss(image_test, latent_test, t)
    print(out.shape)
    print(out)