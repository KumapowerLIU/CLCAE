import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    def __int__(self):
        super(FeatureMatchingLoss, self).__int__()
        self.l1 = nn.L1Loss()

    def forward(self, enc_feat, dec_feat, layer_idx=None):
        loss = []
        if layer_idx is None:
            layer_idx = [i for i in range(len(enc_feat))]
        for i in layer_idx:
            loss.append(self.l1(enc_feat[i], dec_feat[i]))
        return loss
