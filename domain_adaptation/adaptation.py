import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Sequence

class GaussianKernel(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(GaussianKernel, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        x_norm = (x ** 2).sum(1).view(-1, 1)
        dist = x_norm + x_norm.view(1, -1) - 2.0 * torch.mm(x, x.transpose(0, 1))
        return torch.exp(-self.alpha * dist)

class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = self._update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        loss = (kernel_matrix * self.index_matrix).sum() 
        return loss

    def _update_index_matrix(self, batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                            linear: Optional[bool] = True) -> torch.Tensor:
        if index_matrix is None or index_matrix.size(0) != batch_size * 2:
            index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
            if linear:
                for i in range(batch_size):
                    s1, s2 = i, (i + 1) % batch_size
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    index_matrix[s1, s2] = 1. / float(batch_size)
                    index_matrix[t1, t2] = 1. / float(batch_size)
                    index_matrix[s1, t2] = -1. / float(batch_size)
                    index_matrix[s2, t1] = -1. / float(batch_size)
            else:
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:
                            index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                            index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
                for i in range(batch_size):
                    for j in range(batch_size):
                        index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                        index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
        return index_matrix

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReverseLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha

class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int = 256):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.layer3(x)  
        return x

class MultiLinearMap(nn.Module):
    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)

def mmd_loss(features_source, features_target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(features_source.size()[0])
    
    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            bandwidth = torch.clamp(bandwidth, min=1e-6)
            bandwidth = bandwidth / kernel_mul
        
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        
        return sum(kernel_val)
    
    kernels = guassian_kernel(features_source, features_target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    
    XX_diag = torch.diag(XX)
    YY_diag = torch.diag(YY)
    
    XX_sum = (XX.sum() - XX_diag.sum()) / (batch_size * (batch_size - 1))
    YY_sum = (YY.sum() - YY_diag.sum()) / (batch_size * (batch_size - 1))
    XY_sum = XY.sum() / (batch_size * batch_size)
    
    mmd = XX_sum + YY_sum - 2 * XY_sum
    return mmd

def coral_loss(features_source, features_target):
    source_mean = features_source.mean(0, keepdim=True)
    target_mean = features_target.mean(0, keepdim=True)
    
    source_centered = features_source - source_mean
    target_centered = features_target - target_mean
    
    source_cov = torch.mm(source_centered.t(), source_centered) / (features_source.size(0) - 1)
    target_cov = torch.mm(target_centered.t(), target_centered) / (features_target.size(0) - 1)
    
    coral = torch.norm(source_cov - target_cov, p='fro') ** 2
    return coral / (4 * features_source.size(1) ** 2)

def dan_loss(features_source, features_target, args):
    kernels = [GaussianKernel(alpha=2 ** k) for k in range(-5, 5)]
    mkmmd = MultipleKernelMaximumMeanDiscrepancy(kernels, linear=False)
    return mkmmd(features_source, features_target)

def get_adaptation_loss(features_source, features_target, args, **kwargs):
    if args.adaptation == "none":
        return 0.0
    elif args.adaptation == "mmd":
        return mmd_loss(features_source, features_target, 
                       kernel_mul=args.mmd_kernel_mul, 
                       kernel_num=args.mmd_kernel_num, 
                       fix_sigma=args.mmd_fix_sigma)
    elif args.adaptation == "coral":
        return coral_loss(features_source, features_target)
    elif args.adaptation == "dan":
        return dan_loss(features_source, features_target, args)
    else:
        raise ValueError(f"Unsupported adaptation method: {args.adaptation}")