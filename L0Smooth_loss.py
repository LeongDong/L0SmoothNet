import torch
import torch.nn as nn
import torch.nn.functional as F

class L0SmoothLoss(nn.Module):
    def __init__(self):
        super(L0SmoothLoss, self).__init__()

    def forward(self, I, S, lamb, beta):

        loss_sum = torch.FloatTensor(1).cuda().zero_()

        sd = S.detach() #1*1*H*W
        ix = F.pad(sd, pad=[1,0,0,0],mode='reflect') #1*1*H*(W+1)
        iy = F.pad(sd, pad=[0,0,1,0],mode='reflect') #1*1*(H+1)*W

        dx = torch.diff(ix, dim=-1) #1*1*H*W
        dy = torch.diff(iy, dim=-2) #1*1*H*W
        grad = dx * dx + dy * dy #1*1*H*W
        labe = lamb / beta

        h = torch.ones_like(sd)
        v = torch.ones_like(sd)

        h[grad < labe] = 0
        v[grad < labe] = 0
        h = h * dy
        v = v * dx

        Ix = F.pad(S, pad=[1, 0, 0, 0], mode='reflect')  # 1*1*H*(W+1)
        Iy = F.pad(S, pad=[0, 0, 1, 0], mode='reflect')  # 1*1*(H+1)*W
        Dx = torch.diff(Ix, dim=-1) #1*1*H*W
        Dy = torch.diff(Iy, dim=-2) #1*1*H*W

        loss_sum = loss_sum + torch.mean((S - I) * (S - I)) + beta * (torch.mean((Dx - v) * (Dx - v)) + torch.mean((Dy - h) * (Dy - h)))

        return loss_sum