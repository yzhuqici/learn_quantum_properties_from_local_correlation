from torch import nn
import torch
import numpy as np

keys = ['P2x_exps','P2z_exps','Px_exps','Pzxz_exps','correlation_xs','correlation_zs','entropies','RMIs','inter_entropies','target_fidelities']
Heisenberg_keys  = ['P2x_exps','P2z_exps','RMIs','RMI2s']
# Heisenberg_keys  = ['P2x_exps','P2z_exps','RMIs']
twoD_keys = ['expectations','entropies']
cat_keys = ['mpn', 'nprob', 'lw_func','parity']

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

def LossCriterion(outputs, targets):
    loss = 0
    key_loss = {}
    for key in keys:
        key_loss[key] = 0
    for key in keys:
        if key in targets.keys() and key in outputs.keys():
            # print(targets[key].shape)
            # print(outputs[key].shape)
            target =  targets[key].reshape(-1)
            output = outputs[key].reshape(-1)[0:target.shape[0]]
            key_loss[key] += nn.MSELoss()(output,target)
            loss += nn.MSELoss()(output,target)
    return loss, key_loss


def HeisenbergLossCriterion2(outputs, targets):
    loss = 0
    key_loss = {}
    for key in Heisenberg_keys:
        key_loss[key] = 0
    for key in Heisenberg_keys:
        if key in targets.keys() and key in outputs.keys():
            # print(targets[key].shape)
            # print(outputs[key].shape)
            target =  targets[key].reshape(-1)
            output = outputs[key].reshape(-1)[0:target.shape[0]]
            key_loss[key] += LogCoshLoss()(output,target)
            loss += LogCoshLoss()(output,target)
    return loss, key_loss


def TestLossCriterion_ablation(outputs, targets):
    loss = 0
    # keys = ['P2x_exps','P2z_exps','Px_exps','Pzxz_exps','correlation_xs','correlation_zs','entropies','RMIs','inter_entropies']
    keys = ['P2x_exps','P2z_exps']
    # keys = ['entropies']
    key_loss = np.zeros(3)

    for ikey in range(len(keys)):
        key = keys[ikey]
        target = targets[key].reshape(-1)
        output = outputs[key].reshape(-1)[0:target.shape[0]]
        key_loss[ikey] = nn.MSELoss()(output, target)

    # target = targets['target_fidelities'].reshape(-1)
    # output = outputs['target_fidelities'].reshape(-1)[0:target.shape[0]]
    # for i in range(7,11):
    #     loss = nn.MSELoss()(output[i-7], target[i-7])
    #     key_loss[i] = loss
    return key_loss