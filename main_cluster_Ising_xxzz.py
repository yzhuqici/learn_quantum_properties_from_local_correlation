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

from dataset import  StateXXZZData
from model import Backbone, TanH_Decoder,   PrePropertyModel_ablation

num_states = 64*64*16
num_qubits = 9
num_know_mbases = 50
num_measure_qubits = 3
r_dim = 32

test_flag = 1

num_exp_queries = 2
num_ham_queries = 0
num_fidelity_queries = 0
num_entropy_queries = 1
ds = StateXXZZData(num_states=num_states,num_qubits=num_qubits,num_measure_qubits=num_measure_qubits)

train_size = 300
test_size = len(ds) - train_size
torch.manual_seed(10)
train_ds, test_ds = random_split(ds, [train_size, test_size])
train_loader = DataLoader(train_ds,batch_size=1,shuffle=True)
test_loader = DataLoader(ds,batch_size=1)

keys = ['P2x_exps','P2z_exps','entropies']
exp_keys = ['P2x_exps','P2z_exps']
entropy_keys = ['entropies']
fidelity_keys = []

key_settings = '_P2x_P2z_etp'

device_ids=range(torch.cuda.device_count())

bbone = Backbone(x_dim=2**num_measure_qubits, v_dim=2*4**num_measure_qubits, r_dim=r_dim)

exps_decoder = TanH_Decoder(r_dim=r_dim, output_dim=9, task_dim=1)
ham_decoder = None
entropy_decoder = TanH_Decoder(r_dim=r_dim, output_dim=9, task_dim=1)
fidelity_decoder = None

model = PrePropertyModel_ablation(backbone=bbone, exps_decoder=exps_decoder, ham_decoder=ham_decoder, entropy_decoder=entropy_decoder, fidelity_decoder=fidelity_decoder, device_ids=device_ids)
model = nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

lr = 0.0005
epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 10

try:
    model.load_state_dict((torch.load('models/ppmodel_num_qubits9num_measure_qubits3_r_dim32_num_known_mbases50_train_size300_P2x_P2z_etp')))
except:
    print("No Load!")

count = 0

for e in tqdm(range(epochs)):
    test_rs = []
    test_labels = []
    test_phase_values = []
    test_h1h2s = []
    test_entanglement_entropies = []
    test_P2x_exps = []
    test_P2z_exps = []
    test_RMIs = []

    train_rs = []
    train_phase_values = []
    train_h1h2s = []
    random.seed(5)
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

        pred_properties, pred_r = model(x, v, num_exp_queries, num_ham_queries, num_entropy_queries, num_fidelity_queries,exp_keys, entropy_keys)
        test_rs.append(pred_r.cpu().detach().numpy())
        test_phase_values.append(properties['phase_value'].numpy())
        test_h1h2s.append(properties['h1h2'].numpy())

        test_P2x_exps.append(pred_properties['P2x_exps'].cpu().detach().numpy())
        test_P2z_exps.append(pred_properties['P2z_exps'].cpu().detach().numpy())
        test_entanglement_entropies.append(pred_properties['entropies'].cpu().detach().numpy())


    print("train_loss:", end=" ")
    print(train_loss)
    print(train_key_losses)
    print("-------------------------------------")
    print("test_loss:", end=" ")
    print(test_loss)
    print(key_losses)

    if test_flag == 1 :
        key_settings += '_xxzz'
        np.save('results/test_rs_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,np.array(test_rs))
        np.save('results/test_labels_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_labels)
        np.save('results/test_phase_values_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_phase_values)
        np.save('results/test_h1h2s_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_h1h2s)
        np.save('results/test_P2x_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
                test_P2x_exps)
        np.save('results/test_P2z_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
                test_P2z_exps)
        np.save('results/test_entropies_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_entanglement_entropies)
