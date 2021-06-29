import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Hingeloss(_Loss):
    def __init__(self, margin=1., reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        # self.hinge = nn.HingeEmbeddingLoss(margin=margin, reduction='mean')

    def forward(self, x, label):
        # hinge_label = (- torch.ones_like(x)).scatter(dim=1, index=label.view(-1, 1), src=torch.ones_like(x))
        # print(hinge_label)
        # loss = self.hinge(x, hinge_label)
        gt = x * torch.zeros_like(x).scatter(dim=1, index=label.view(-1, 1), src=torch.ones_like(x))
        loss = ((self.margin - x).clamp_min(min=0) * torch.ones_like(x).scatter(dim=1, index=label.view(-1, 1), src=torch.zeros_like(x)) + gt)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class PointLoss(_Loss):
    """
    Calculate error between the predicted temperature field and real field over monitoring points
    """
    def __init__(self):
        super().__init__()

    def forward(self, monitor_y, layout_pred, criterion = torch.nn.L1Loss()):

        ones = torch.ones_like(monitor_y).to(device=layout_pred.device)
        zeros = torch.zeros_like(monitor_y).to(device=layout_pred.device)
        ind = torch.where(monitor_y>0, ones, zeros)
        return criterion(monitor_y, layout_pred * ind)


class TVLoss(_Loss):
    """
    Calculate one-order gradient information of predicted temperature field
    """
    def __init__(self):
        super().__init__()

    def forward(self, layout_pred):
        batch_size = layout_pred.size()[0]
        h_x = layout_pred.size()[2]
        w_x = layout_pred.size()[3]
        count_h = self._tensor_size(layout_pred[:,:,1:,:]) 
        count_w = self._tensor_size(layout_pred[:,:,:,1:])
        h_tv = torch.pow((layout_pred[:,:,1:,:] - layout_pred[:,:,:h_x-1,:]),2).sum()  
        w_tv = torch.pow((layout_pred[:,:,:,1:] - layout_pred[:,:,:,:w_x-1]), 2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


#---------------------------------------Laplace loss-------------------------------------------------------------------------------------
class LaplaceLoss(_Loss):
    """
    Calculate second-order gradient information of predicted temperature field
    """
    def __init__(
        self, base_loss=MSELoss(reduction='mean'), nx=200,
        length=0.1, weight=[[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], bcs=None, margin=1, hinge_reduction='mean'
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weight = torch.Tensor(weight)
        self.bcs = bcs
        self.length = length
        self.nx = nx
        self.margin = margin
        self.hinge_reduction = hinge_reduction
        self.scale_factor = 1                       # self.nx/200
        TEMPER_COEFFICIENT = 50.0
        STRIDE = self.length / self.nx
        self.cof = -1 * STRIDE**2/TEMPER_COEFFICIENT

    def laplace(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat):
        N, C, W, H = layout.shape
        layout = interpolate(layout, scale_factor=self.scale_factor)
        heat = pad(heat, [1, 1, 1, 1], mode='replicate')    # constant, reflect, replicate
        layout_pred = self.laplace(heat) 

        ones = torch.ones_like(layout).to(device=layout.device)
        zeros = torch.zeros_like(layout).to(device=layout.device)
        ind = torch.where(layout>1e-2, zeros, ones)

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            return self.base_loss(layout_pred[..., 1:-1, 1:-1] * ind, self.cof * layout[..., 1:-1, 1:-1] * ind) 
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    layout_pred[..., idx_start:idx_end, :1] = self.cof * layout[..., idx_start:idx_end, :1]
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    layout_pred[..., idx_start:idx_end, -1:] = self.cof * layout[..., idx_start:idx_end, -1:]
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., :1, idx_start:idx_end] = self.cof * layout[..., :1, idx_start:idx_end]
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., -1:, idx_start:idx_end] = self.cof * layout[..., -1:, idx_start:idx_end]
                else:
                    raise ValueError("bc error!")
            return self.base_loss(layout_pred * ind, self.cof * layout * ind) 


class OutsideLoss(_Loss):
    """
    Calculate prediction information over the boundaries
    """
    def __init__(
        self, base_loss=MSELoss(reduction='mean'), length=0.1, u_D=298, bcs=None, nx=21
    ):
        super().__init__()
        self.base_loss = base_loss
        self.u_D = u_D
        self.slice_bcs = []
        self.bcs = bcs
        self.nx = nx
        self.length = length

    def forward(self, x):
        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all bcs are Dirichlet
            d1 = x[:, :, :1, :]
            d2 = x[:, :, -1:, :]
            d3 = x[:, :, 1:-1, :1]
            d4 = x[:, :, 1:-1, -1:]
            point = torch.cat([d1.flatten(), d2.flatten(), d3.flatten(), d4.flatten()], dim=0)
            return self.base_loss(point, torch.zeros_like(point))
        loss = 0
        for bc in self.bcs:
            if bc[0][1] == 0 and bc[1][1] == 0:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, :1]
                loss += self.base_loss(point, torch.zeros_like(point))
            elif bc[0][1] == self.length and bc[1][1] == self.length:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, -1:]
                loss += self.base_loss(point, torch.zeros_like(point))
            elif bc[0][0] == 0 and bc[1][0] == 0:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., :1, idx_start:idx_end]
                loss += self.base_loss(point, torch.zeros_like(point))
            elif bc[0][0] == self.length and bc[1][0] == self.length:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., -1:, idx_start:idx_end]
                loss += self.base_loss(point, torch.zeros_like(point))
            else:
                raise ValueError("bc error!")
        return loss