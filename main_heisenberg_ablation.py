import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import pickle

from dataset import HeisenbergData
from model import Backbone, TanH_Decoder,  PreHeisenbergPropertyModel, NoAct_Decoder
from util import HeisenbergLossCriterion2

num_states = 441
num_qubits = 50
num_know_mbases = 200
num_measure_qubits = 3
r_dim = 24

test_flag = 1

num_exp_queries = 2
num_ham_queries = 0
num_fidelity_queries = 0
num_entropy_queries = 2

ds = HeisenbergData(num_states=num_states,num_qubits=num_qubits,num_measure_qubits=num_measure_qubits)
train_size = 80
test_size = len(ds) - train_size
torch.manual_seed(42)
train_ds, test_ds = random_split(ds, [train_size, test_size])
train_loader = DataLoader(train_ds,batch_size=1,shuffle=True)
test_loader = DataLoader(ds,batch_size=1)

keys = ['P2x_exps','P2z_exps','RMIs','RMI2s']
# keys = ['P2x_exps','P2z_exps']
# keys = ['RMIs','RMI2s']
exp_keys = ['P2x_exps','P2z_exps']
# exp_keys = []
entropy_keys = ['RMIs','RMI2s']
# entropy_keys = []
# entropy_keys = ['RMI2s']


key_settings = 'model2'
key_settings += '_P2x_P2z_RMI1_RMI2'
# key_settings += '_P2x_P2z_RMI1'
# key_settings += '_P2x_P2z'
# key_settings += '_RMI1_RMI2'

device_ids=range(torch.cuda.device_count())
bbone = Backbone(x_dim=2**num_measure_qubits, v_dim=2*4**num_measure_qubits, r_dim=r_dim)
exps_decoder = TanH_Decoder(r_dim=r_dim, output_dim=100, task_dim=1)
# exps_decoder = None
ham_decoder = None
entropy_decoder = NoAct_Decoder(r_dim=r_dim, output_dim=6, task_dim=1)
# entropy_decoder = None
fidelity_decoder = None

model = PreHeisenbergPropertyModel(backbone=bbone, exps_decoder=exps_decoder, ham_decoder=ham_decoder, entropy_decoder=entropy_decoder, fidelity_decoder=fidelity_decoder, device_ids=device_ids)
model = nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

lr = 0.0001
epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# batch_size = 5
batch_size = 2


try:
    model.load_state_dict(torch.load(
        'models/p_heisenberg_model_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings))
except:
    print("No Load!")

# torch.save(model.module.backbone.state_dict(), 'models/heisenberg_rep_model_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
#             num_measure_qubits) + '_r_dim' + str(
#             r_dim) + '_num_known_mbases' + str(num_know_mbases) +key_settings)

count = 0

for e in tqdm(range(epochs)):
    test_rs = []
    test_labels = []
    test_phase_values = []
    test_h1h2s = []
    train_rs = []
    train_phase_values = []
    train_h1h2s = []
    test_RMI1s = []
    test_RMI2s = []
    test_P2xs = []
    test_P2zs = []
    random.seed(42)
    torch.manual_seed(42)
    index = 0
    loss = 0
    train_loss = 0
    test_loss = 0
    train_key_losses = {}
    all_test_key_losses = []

    key_losses = {}
    for key in keys:
        train_key_losses[key] = 0
        key_losses[key] = 0

    if test_flag == 0:
        for mbases, mresults, properties in train_loader:
            count += 1
            index += 1
            mbases = mbases.cuda(device=device_ids[0])
            mresults = mresults.cuda(device=device_ids[0])
            b, m, *x_dims = mresults.shape
            _, _, *v_dims = mbases.shape

            indices = list(range(0, m))
            random.shuffle(indices)
            representation_idx = indices[0:num_know_mbases]

            x = mresults[:, representation_idx, :].float()
            v = mbases[:, representation_idx, :].float()

            pred_properties, pred_r = model(x, v, num_exp_queries, num_ham_queries, num_entropy_queries, num_fidelity_queries)
            train_rs.append(pred_r.cpu().detach().numpy())
            train_phase_values.append(properties['phase_value'].numpy())
            train_h1h2s.append(properties['jpdelta'].numpy())

            if index <= 40:
                known_tgts = {}
                for ik in range(0,len(exp_keys)):
                    tmp_key = exp_keys[ik]
                    known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()
            elif index <= 80:
                known_tgts = {}
                for ik in range(0, len(entropy_keys)):
                    tmp_key = entropy_keys[ik]
                    known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()

            total_loss, key_loss = HeisenbergLossCriterion2(pred_properties, known_tgts)
            loss += total_loss
            train_loss += total_loss
            for key in keys:
                train_key_losses[key] += key_loss[key]

            if count % batch_size == 0:
                if loss == 0:
                    pass
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss = 0

    for mbases, mresults, properties in test_loader:
        count += 1
        mbases = mbases.cuda(device=device_ids[0])
        mresults = mresults.cuda(device=device_ids[0])
        b, m, *x_dims = mresults.shape
        _, _, *v_dims = mbases.shape

        indices = list(range(0, m))
        random.shuffle(indices)
        representation_idx = indices[0:num_know_mbases]

        x = mresults[:, representation_idx, :].float()
        v = mbases[:, representation_idx, :].float()

        pred_properties, pred_r = model(x, v, num_exp_queries, num_ham_queries, num_entropy_queries, num_fidelity_queries)
        test_rs.append(pred_r.cpu().detach().numpy())
        test_phase_values.append(properties['phase_value'].numpy())
        test_h1h2s.append(properties['jpdelta'].numpy())
        test_RMI1s.append(pred_properties['RMIs'].cpu().detach().numpy())
        test_RMI2s.append(pred_properties['RMI2s'].cpu().detach().numpy())
        test_P2xs.append(pred_properties['P2x_exps'].cpu().detach().numpy())
        test_P2zs.append(pred_properties['P2z_exps'].cpu().detach().numpy())

        known_tgts = {}
        key_list = list(keys)
        for ik in range(0, len(key_list)):
            tmp_key = key_list[ik]
            known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()

        total_loss, key_loss = HeisenbergLossCriterion2(pred_properties, known_tgts)

        for key in keys:
            key_losses[key] += key_loss[key]

        test_loss += total_loss


    print("train_loss:", end=" ")
    print(train_loss)
    print(train_key_losses)
    print("-------------------------------------")
    print("test_loss:", end=" ")
    print(test_loss)
    print(key_losses)

    if test_flag == 0:
        torch.save(model.state_dict(), 'models/p_heisenberg_model_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings)
