#######################################################################################
# Author: Wei-Cheng Tseng from https://github.com/WeiChengTseng/Pytorch-PCGrad/tree/master
# Script: pcgrad.py
# Function: Pytorch implementation of Gradient surgery for multi-task learning 
#           from paper by Yu, T, Kumar, S, Gupta, A, and Levine, S and Hausman, K
# Date: 02/06/2026
#######################################################################################
from seed_utils import set_seed

set_seed(42)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import torch.optim as optim

#######################################################################################
class PCGrad():
    """
    PyTorch implementation of Gradient Surgery for Multi-Task Learning.
    
    Based on "Gradient Surgery for Multi-Task Learning" by Yu et al. (2020).
    Projects conflicting gradients to reduce negative transfer between tasks.
    
    Reference: https://github.com/WeiChengTseng/Pytorch-PCGrad/tree/master
    """
    def __init__(self, optimizer, reduction='mean'):
        """
        Initialize PCGrad optimizer wrapper.
        
        Args:
            optimizer (torch.optim.Optimizer): Base optimizer to wrap.
            reduction (str): How to combine gradients ('mean' or 'sum'). Defaults to 'mean'.
        """
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        """
        Get the underlying optimizer.
        
        Returns:
            torch.optim.Optimizer: The wrapped optimizer.
        """
        return self._optim

    def zero_grad(self):
        """
        Clear the gradient of all parameters.
        
        Returns:
            None
        """

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        Update the parameters with the modified gradient.
        
        Returns:
            Result from optimizer step.
        """

        return self._optim.step()

    def pc_backward(self, objectives):
        """
        Calculate gradients and apply gradient surgery to resolve conflicts.
        
        Projects conflicting gradients (negative cosine similarity) to prevent
        interference between objectives.
        
        Args:
            objectives (list): List of loss tensors, one per objective.
            
        Returns:
            None: Modifies parameter gradients in-place.
        """

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        """
        Project gradients to resolve conflicts between objectives.
        
        For each pair of gradients with negative dot product, projects the first
        gradient away from the second to eliminate the conflicting component.
        
        Args:
            grads (list): List of gradient tensors.
            has_grads (list): List of masks indicating which parameters have gradients.
            shapes (list, optional): Parameter shapes. Defaults to None.
            
        Returns:
            torch.Tensor: Merged gradient after conflict resolution.
        """
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        if merged_grad.isnan().any():
            merged_grad = torch.stack([g for g in grads]).sum(dim=0)
            if merged_grad.isnan().any():
                raise ValueError("Merged grad has NaN even after fallback")
        return merged_grad

    def _set_grad(self, grads):
        """
        Set the modified gradients to the network parameters.
        
        Args:
            grads (list): List of gradient tensors to assign to parameters.
            
        Returns:
            None
        """

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        """
        Pack the gradients of the network parameters for each objective.
        
        Computes gradients for each objective separately and flattens them into
        vectors for gradient surgery operations.
        
        Args:
            objectives (list): List of loss tensors.
            
        Returns:
            tuple: (grads, shapes, has_grads) where:
                - grads: List of flattened gradient vectors
                - shapes: List of parameter shapes
                - has_grads: List of masks for parameters with gradients
        """
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)
