import torch
import torch.nn as nn
import torch.nn.functional as F




def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss_func = nn.L1Loss()

    def forward(self, x, y):
        loss = self.loss_func(x, y)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, x, y):
        loss = self.loss_func(x, y)
        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, x, y):
        loss = self.loss_func(x, y)
        return loss

class L1MSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(L1MSELoss, self).__init__()
        self.loss_func_l1 = nn.L1Loss()
        self.loss_func_mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, x, y):
        loss = self.alpha * self.loss_func_mse(x, y) + (1- self.alpha)*self.loss_func_l1(x, y)
        return loss

class MSETVLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MSETVLoss, self).__init__()
        self.loss_func_mse = nn.MSELoss()
        self.loss_func_tv = TVLoss()
        self.alpha = alpha

    def forward(self, x, y):
        loss = self.alpha * self.loss_func_mse(x, y) + (1- self.alpha)*self.loss_func_tv(y)
        return loss

class L1TVLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(L1TVLoss, self).__init__()
        self.loss_func_l1 = nn.L1Loss()
        self.loss_func_tv = TVLoss()
        self.alpha = alpha

    def forward(self, x, y):
        loss = self.alpha * self.loss_func_l1(x, y) + (1- self.alpha)*self.loss_func_tv(y)
        return loss

class SSLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(SSLoss, self).__init__()
        self.loss_func_mse = nn.MSELoss()
        self.loss_func_l1 = nn.L1Loss()
        self.loss_func_char = CharbonnierLoss()
        self.alpha = alpha

    def forward(self, x, y, z):
        loss = self.alpha*self.loss_func_char(x, y) + (1-self.alpha)*self.loss_func_l1(z, y)

        return loss


def l1_loss(x,y_out):
    loss = L1Loss()(x, y_out)
    return loss

def mse_loss(x,y_out):
    loss = MSELoss()(x, y_out)
    return loss

def l1mse_loss(x,y_out, alpha=0.5):
    loss = alpha*MSELoss()(x, y_out)+ (1-alpha)*L1Loss()(x, y_out)
    return loss

def msetv_loss(x,y_out, alpha=0.5):
    loss = alpha * MSELoss()(x, y_out) + (1 - alpha) * TVLoss()(y_out)
    return loss

def sl1_loss(x,y_out):
    loss = SmoothL1Loss()(x, y_out)
    return loss