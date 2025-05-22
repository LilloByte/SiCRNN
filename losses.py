import torch
import torch.nn.functional as F


# Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.m = margin

    def forward(self, output1, output2, label):
        # euclidean distance
        e_d = F.pairwise_distance(output1, output2)
        # contrastive loss
        c_loss = torch.mean((1-label) * torch.pow(e_d, 2) + (label) * torch.pow(torch.clamp(self.m - e_d, min=0.0), 2))

        return c_loss