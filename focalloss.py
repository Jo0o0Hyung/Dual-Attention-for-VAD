import torch
import torch.nn as nn


class focal_loss(nn.Module):
    def __init__(self, alpha, gamma):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        BCloss = self.BCE(input=logits, target=labels.float())

        if self.gamma == 0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCloss

        weighted_loss = self.alpha * loss
        fl = torch.sum(weighted_loss)

        fl /= torch.sum(labels)

        return fl
