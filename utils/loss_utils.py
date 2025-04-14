import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        self.para_dict = para_dict
        self.num_classes = self.para_dict["num_classes"]
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']
        if 'BalancedSoftmaxCE_T' in self.para_dict.keys():
            self.Temperature = self.para_dict['BalancedSoftmaxCE_T']
        else:
            self.Temperature = 1

        self.weight_list = None

        #settings of defferred re-balancing by re-weighting (DRW)
        self.drw = self.para_dict['two_stage_train_flag']
        self.drw_start_epoch = self.para_dict['two_stage_train_epoch']

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        loss = F.cross_entropy(inputs, targets, weight=self.weight_list)
        return loss

    def update(self, epoch):
        """
        Adopt cost-sensitive cross-entropy as the default
        Args:
            epoch: int. starting from 1.
        """
        pass

class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.weight=weight

    def forward(self, x, y, **kwargs):
        x = torch.log_softmax(x, dim=1)
        print(type(y))
        print(type(torch.range(0, x.shape[0]-1).long()))
        loss = -x[torch.range(0, x.shape[0]-1).long(), y] * self.weight[y]
        loss = loss.sum()/self.weight[y].sum()
        return loss


class BalancedSoftmaxCE(CrossEntropy):
    r"""
    References:
    Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.

    Equation: Loss(x, c) = -log(\frac{n_c*exp(x)}{sum_i(n_i*exp(i)})
    """

    def __init__(self, para_dict=None):
        super(BalancedSoftmaxCE, self).__init__(para_dict)
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)

    def forward(self, inputs, targets,  **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        logits = inputs + self.weight_list.unsqueeze(0).expand(inputs.shape[0], -1).log()
        loss = F.cross_entropy(input=logits, target=targets)
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.bsce_weight
        else:
            self.weight_list = torch.ones(self.bsce_weight.shape).to(self.device)
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.bsce_weight
        
        self.weight_list = torch.pow(self.weight_list,self.Temperature)