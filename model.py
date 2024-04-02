import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Backbone(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(Backbone, self).__init__()
        # Final representation size
        self.r_dim = k = r_dim
        self.v_dim = v_dim
        self.linear1 = nn.Linear(v_dim, k)
        self.linear2 = nn.Linear(x_dim, k)
        self.linear3 = nn.Linear(3, k)
        self.linear4 = nn.Linear(2*k, k)
        self.linear5 = nn.Linear(k , k)
        # self.linear6 = nn.Linear(2*k, k)


    def forward(self, x, v):
        v1 = F.relu(self.linear1(v[:,0:self.v_dim]))
        x = F.relu(self.linear2(x))
        v2 = F.relu(self.linear3(v[:,self.v_dim:]))
        merge = torch.cat([v1,v2], dim=1)
        rv = F.relu(self.linear4(merge))
        rx = F.relu(self.linear5(x))
        return rx, rv
class TanH_Decoder(nn.Module):
    def __init__(self, r_dim, output_dim, task_dim):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(TanH_Decoder, self).__init__()
        # Final representation size
        k = r_dim
        self.linear1 = nn.Linear(r_dim, k)
        self.linear2 = nn.Linear(task_dim, k)
        self.linear3 = nn.Linear(2*k, k)
        self.linear4 = nn.Linear(k, k)
        self.linear5 = nn.Linear(k,output_dim)


    def forward(self, r, t):
        r = F.relu(self.linear1(r))
        t = F.relu(self.linear2(t))
        merge = torch.cat([r,t], dim=1)
        r = F.relu(self.linear3(merge))
        r = F.relu(self.linear4(r))
        r = F.tanh(self.linear5(r))
        return r
class NoAct_Decoder(nn.Module):
    def __init__(self, r_dim, output_dim, task_dim):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(NoAct_Decoder, self).__init__()
        # Final representation size
        k = r_dim
        self.linear1 = nn.Linear(r_dim, k)
        self.linear2 = nn.Linear(task_dim, k)
        self.linear3 = nn.Linear(2*k, k)
        self.linear4 = nn.Linear(k, k)
        self.linear5 = nn.Linear(k ,output_dim)


    def forward(self, r, t):
        r = F.relu(self.linear1(r))
        t = F.relu(self.linear2(t))
        merge = torch.cat([r,t], dim=1)
        r = F.relu(self.linear3(merge))
        r = F.relu(self.linear4(r))
        r = self.linear5(r)
        return r
class PrePropertyModel_ablation(nn.Module):
    def __init__(self, backbone, exps_decoder, ham_decoder=None, entropy_decoder=None, fidelity_decoder=None, device_ids=None):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(PrePropertyModel_ablation, self).__init__()
        # Final representation size
        self.backbone = backbone
        self.exps_decoder = exps_decoder
        self.ham_decoder = ham_decoder
        self.entropy_decoder = entropy_decoder
        self.fidelity_decoder = fidelity_decoder
        self.device_ids = device_ids

    def forward(self, x, v, num_exp_queries, num_ham_queries, num_entropy_queries, num_fidelity_queries, exp_keys, entropy_keys, fidelity_keys=None):
        b, m, *x_dims = x.shape
        _, _, *v_dims = v.shape

        x = x.view((-1,*x_dims))
        v = v.view((-1, *v_dims))

        rx, rv = self.backbone(x, v)
        rx = rx.view((b, m, -1))
        rv = rv.view((b, m, -1))
        r = rx + rv
        r = torch.sum(r, dim=1)
        # print(r.shape)

        pred_exps = []
        for index in range(num_exp_queries):
            task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
            pred = self.exps_decoder(r, task_index)
            pred_exps.append(pred)

        pred_hs = []
        if self.ham_decoder != None:
            for index in range(num_ham_queries):
                task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
                pred = self.ham_decoder(r, task_index)
                pred_hs.append(pred)

        pred_entropies = []
        if self.entropy_decoder != None:
            for index in range(num_entropy_queries):
                task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
                pred = self.entropy_decoder(r, task_index)
                pred_entropies.append(pred)

        pred_fidelities = []
        if self.fidelity_decoder != None:
            for index in range(num_fidelity_queries):
                task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
                pred = self.fidelity_decoder(r, task_index)
                pred_fidelities.append(pred)

        out = {}
        for i in range(num_exp_queries):
            out[exp_keys[i]] = pred_exps[i]

        if self.ham_decoder != None:
            out['h1'] = pred_hs[0] * 0.8 + 0.8
            out['h2'] = pred_hs[1] * 1.6

        if self.entropy_decoder != None:
            for i in range(num_entropy_queries):
                out[entropy_keys[i]] = pred_entropies[i] + 1
            # out['entropies'] = pred_entropies[0] + 1
            # out['RMIs'] = pred_entropies[1] + 1
            # out['inter_entropies'] = pred_entropies[2]*2 + 2

        if self.fidelity_decoder != None:
            out['target_fidelities'] = pred_fidelities[0]


        return out, r
class PreHeisenbergPropertyModel(nn.Module):
    def __init__(self, backbone, exps_decoder, ham_decoder=None, entropy_decoder=None, fidelity_decoder=None, device_ids=None):
        """
        A neural network with skip connections which converts the input observable and
        its corresponding expectation into a representation which is more dense than the input.
        """
        super(PreHeisenbergPropertyModel, self).__init__()
        # Final representation size
        self.backbone = backbone
        self.exps_decoder = exps_decoder
        self.ham_decoder = ham_decoder
        self.entropy_decoder = entropy_decoder
        self.fidelity_decoder = fidelity_decoder
        self.device_ids = device_ids

    def forward(self, x, v, num_exp_queries, num_ham_queries, num_entropy_queries, num_fidelity_queries):
        b, m, *x_dims = x.shape
        _, _, *v_dims = v.shape

        x = x.view((-1,*x_dims))
        v = v.view((-1, *v_dims))

        rx, rv = self.backbone(x, v)
        rx = rx.view((b, m, -1))
        rv = rv.view((b, m, -1))
        r = rx + rv
        r = torch.sum(r, dim=1)
        # print(r.shape)

        pred_exps = []
        if self.exps_decoder != None:
            for index in range(num_exp_queries):
                task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
                pred = self.exps_decoder(r, task_index)
                pred_exps.append(pred)

        pred_hs = []
        if self.ham_decoder != None:
            for index in range(num_ham_queries):
                task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
                pred = self.ham_decoder(r, task_index)
                pred_hs.append(pred)

        pred_entropies = []
        if self.entropy_decoder != None:
            for index in range(num_entropy_queries):
                task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
                pred = self.entropy_decoder(r, task_index)
                pred_entropies.append(pred)

        pred_fidelities = []
        if self.fidelity_decoder != None:
            for index in range(num_fidelity_queries):
                task_index = torch.Tensor(np.ones((b,1))*index).cuda(device=self.device_ids[0]).float()
                pred = self.fidelity_decoder(r, task_index)
                pred_fidelities.append(pred)

        if self.exps_decoder != None:
            out = {
                'P2x_exps': pred_exps[0],
                'P2z_exps': pred_exps[1],
                # 'P2x_exps': pred_exps[0]*2-1,
                # 'P2z_exps': pred_exps[1]*2-1
                # 'correlation_xs': pred_exps[2],
                # 'correlation_zs': pred_exps[3]
            }
        else:
            out = {}

        if self.ham_decoder != None:
            out['h1'] = pred_hs[0] * 0.8 + 0.8
            out['h2'] = pred_hs[1] * 1.6

        if self.entropy_decoder != None:
            # out['entropies'] = pred_entropies[0] + 1
            out['RMIs'] = pred_entropies[0] + 1
            out['RMI2s'] = pred_entropies[1] + 1
            # out['inter_entropies'] = pred_entropies[2]*2 + 2

        if self.fidelity_decoder != None:
            out['fidelities'] = pred_fidelities[0]

        return out, r


