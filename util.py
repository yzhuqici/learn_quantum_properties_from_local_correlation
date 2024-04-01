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

class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)

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

def HeisenbergLossCriterion(outputs, targets):
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

# def HeisenbergLossCriterion(outputs, targets):
#     loss = 0
#     key_loss = {}
#     for key in Heisenberg_keys:
#         key_loss[key] = 0
#     for key in Heisenberg_keys:
#         if key in targets.keys() and key in outputs.keys():
#             # print(targets[key].shape)
#             # print(outputs[key].shape)
#             target =  targets[key].reshape(-1)
#             output = outputs[key].reshape(-1)[0:target.shape[0]]
#             key_loss[key] += nn.MSELoss()(output,target)
#             loss += nn.MSELoss()(output,target)
#     return loss, key_loss

def twoDLossCriterion(outputs, targets):
    loss = 0
    key_loss = {}
    # print(outputs.keys())
    # print(targets.keys())
    for key in twoD_keys:
        key_loss[key] = 0
    for key in twoD_keys:
        if key in targets.keys() and key in outputs.keys():
            # print(targets[key].shape)
            # print(outputs[key].shape)
            target = targets[key].reshape(-1)
            output = outputs[key].reshape(-1)[0:target.shape[0]]
            # print(target)
            # print(output)
            key_loss[key] += nn.MSELoss()(output,target)
            loss += nn.MSELoss()(output,target)
    return loss, key_loss

def CatLossCriterion(outputs, targets):
    loss = 0
    key_loss = {}
    # print(outputs.keys())
    # print(targets.keys())
    for key in cat_keys:
        key_loss[key] = 0
    for key in cat_keys:
        if key in targets.keys() and key in outputs.keys():
            # print(targets[key].shape)
            # print(outputs[key].shape)
            target =  targets[key].reshape(-1)
            output = outputs[key].reshape(-1)[0:target.shape[0]]
            # print(target)
            # print(output)
            key_loss[key] += nn.MSELoss()(output,target)
            loss += nn.MSELoss()(output,target)
    return loss, key_loss

def TestLossCriterion(outputs, targets):
    loss = 0
    keys = ['Px_exps','Pzxz_exps','correlation_xs','correlation_zs','entropies','RMIs','inter_entropies']
    key_loss = np.zeros(11)

    for ikey in range(len(keys)):
        key = keys[ikey]
        target = targets[key].reshape(-1)
        output = outputs[key].reshape(-1)[0:target.shape[0]]
        key_loss[ikey] = nn.MSELoss()(output, target)

    target = targets['target_fidelities'].reshape(-1)
    output = outputs['target_fidelities'].reshape(-1)[0:target.shape[0]]
    for i in range(7,11):
        loss = nn.MSELoss()(output[i-7], target[i-7])
        key_loss[i] = loss
    return key_loss

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

def HeisenbergTestLossCriterion(outputs, targets):
    keys = ['RMIs','RMI2s','P2x_exps', 'P2z_exps']
    # keys = ['RMIs', 'P2x_exps', 'P2z_exps']
    key_loss = np.zeros(7)
    for ikey in range(len(keys)):
        key = keys[ikey]
        target = targets[key].reshape(-1)
        output = outputs[key].reshape(-1)[0:target.shape[0]]
        key_loss[ikey] = nn.MSELoss()(output, target)
        if key == 'P2x_exps':
            target = targets[key].reshape(-1)[0:49]
            output = outputs[key].reshape(-1)[0:target.shape[0]]
            key_loss[3] = nn.MSELoss()(output, target)
            target = targets[key].reshape(-1)[49:]
            output = outputs[key].reshape(-1)[49:target.shape[0]+49]
            key_loss[4] = nn.MSELoss()(output, target)
        elif key == 'P2z_exps':
            target = targets[key].reshape(-1)[0:49]
            output = outputs[key].reshape(-1)[0:target.shape[0]]
            key_loss[5] = nn.MSELoss()(output, target)
            target = targets[key].reshape(-1)[49:]
            output = outputs[key].reshape(-1)[49:target.shape[0]+49]
            key_loss[6] = nn.MSELoss()(output, target)

    return key_loss

def twoDTestLossCriterion(outputs, targets):
    keys = ['expectations', 'entropies']
    key_loss = np.zeros(2)
    for ikey in range(len(keys)):
        key = keys[ikey]
        target = targets[key].reshape(-1)
        output = outputs[key].reshape(-1)[0:target.shape[0]]
        key_loss[ikey] = nn.MSELoss()(output, target)
    return key_loss

def CatTestLossCriterion(outputs, targets):
    # keys = ['mpn','nprob','lw_func','parity']
    keys = ['mpn', 'nprob', 'lw_func']
    key_loss = np.zeros(3)
    for ikey in range(len(keys)):
        key = keys[ikey]
        target = targets[key].reshape(-1)
        output = outputs[key].reshape(-1)[0:target.shape[0]]
        key_loss[ikey] = nn.MSELoss()(output, target)
    return key_loss