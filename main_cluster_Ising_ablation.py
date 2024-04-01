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

from dataset import StateData
from model2 import Backbone, TanH_Decoder, Relu_Decoder, Sigmoid_Decoder, PrePropertyModel_ablation
from util import LossCriterion, TestLossCriterion, TestLossCriterion_ablation

num_states = 4096
num_qubits = 9
num_know_mbases = 50
num_measure_qubits = 3
r_dim = 32

test_flag = 1

num_exp_queries = 2
num_ham_queries = 0
num_fidelity_queries = 0
num_entropy_queries = 1

ds = StateData(num_states=num_states,num_qubits=num_qubits,num_measure_qubits=num_measure_qubits)
train_size = 300 #int(0.2 * len(ds))
test_size = len(ds) - train_size
torch.manual_seed(10)
# generator1 = torch.Generator().manual_seed(10)
train_ds, test_ds = random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(3407))
print(len(train_ds))
train_loader = DataLoader(train_ds,batch_size=1,shuffle=True)
test_loader = DataLoader(ds,batch_size=1)

# keys = ['Px_exps','Pzxz_exps','correlation_xs','correlation_zs','entropies','RMIs','inter_entropies','target_fidelities']
# exp_keys = ['Px_exps','Pzxz_exps','correlation_xs','correlation_zs']
# entropy_keys = ['entropies','RMIs','inter_entropies']
# fidelity_keys = ['target_fidelities']

keys = ['P2x_exps','P2z_exps','entropies']
# keys = ['entropies']
# exp_keys = []
exp_keys = ['P2x_exps','P2z_exps']
# entropy_keys = []
entropy_keys = ['entropies']
# fidelity_keys = ['target_fidelities']

# key_settings = '_P2x_P2z'
# key_settings = '_etp'
key_settings = '_P2x_P2z_etp'

device_ids=range(torch.cuda.device_count())
bbone = Backbone(x_dim=2**num_measure_qubits, v_dim=2*4**num_measure_qubits, r_dim=r_dim)
exps_decoder = TanH_Decoder(r_dim=r_dim, output_dim=num_qubits, task_dim=1)
ham_decoder = None
entropy_decoder = TanH_Decoder(r_dim=r_dim, output_dim=num_qubits, task_dim=1)
fidelity_decoder = None

model = PrePropertyModel_ablation(backbone=bbone, exps_decoder=exps_decoder, ham_decoder=ham_decoder, entropy_decoder=entropy_decoder, fidelity_decoder=fidelity_decoder, device_ids=device_ids)
model = nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

lr = 0.0001
epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 10
best_loss = 30
best_test_loss = 0.1

try:
    model.load_state_dict(torch.load(
        'models/ppmodel_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings))
    # model.load_state_dict((torch.load('models/ppmodel_num_qubits9num_measure_qubits3_r_dim32_num_known_mbases50_train_size300')))
except:
    print("No Load!")




count = 0

for e in tqdm(range(epochs)):
    test_rs = []
    test_labels = []
    test_phase_values = []
    test_h1h2s = []
    test_entanglement_entropies = []
    # test_target_fidelites = []
    # test_Px_exps = []
    # test_Pzxz_exps = []
    test_P2x_exps = []
    test_P2z_exps = []
    # test_correlation_xs = []
    # test_correlation_zs = []
    # test_RMIs = []
    # test_inter_entropies = []

    train_rs = []
    train_phase_values = []
    train_h1h2s = []
    random.seed(5)
    # random.seed(8)
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
            # print(index)
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

            pred_properties, pred_r = model(x, v, num_exp_queries, num_ham_queries, num_entropy_queries, num_fidelity_queries,exp_keys, entropy_keys)
            train_rs.append(pred_r.cpu().detach().numpy())
            train_phase_values.append(properties['phase_value'].numpy())
            train_h1h2s.append(properties['h1h2'].numpy())

            # key_list = list(properties.keys())
            # indices2 = list(range(0, len(key_list)))
            # random.shuffle(indices2)
            # num_known_property = random.randint(2, len(key_list))
            # known_tgts = {}
            # for ik in range(0,num_known_property):
            #     tmp_key = key_list[indices2[ik]]
            #     known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()

            if index <= 300:
                known_tgts = {}
                for ik in range(0,len(exp_keys)):
                    tmp_key = exp_keys[ik]
                    known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()
            elif index <= 0:
                known_tgts = {}
                for ik in range(0, len(entropy_keys)):
                    tmp_key = entropy_keys[ik]
                    known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()
            # else:
            #     known_tgts = {}
            #     for ik in range(0, len(fidelity_keys)):
            #         tmp_key = fidelity_keys[ik]
            #         known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()

            total_loss, key_loss = LossCriterion(pred_properties, known_tgts)
            # print(total_loss)
            # print(pred_properties)
            loss += total_loss
            train_loss += total_loss
            for key in keys:
                train_key_losses[key] += key_loss[key]

            if count % batch_size == 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0

        np.save('results/train_rs_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, np.array(train_rs))
        np.save('results/train_phase_values_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, train_phase_values)
        np.save('results/train_h1h2s_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, train_h1h2s)

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
        test_labels.append(properties['phase_label'].numpy())
        test_phase_values.append(properties['phase_value'].numpy())
        test_h1h2s.append(properties['h1h2'].numpy())

        # test_Px_exps.append(properties['Px_exps'].numpy())
        # test_Pzxz_exps.append(properties['Pzxz_exps'].numpy())
        test_P2x_exps.append(pred_properties['P2x_exps'].cpu().detach().numpy())
        test_P2z_exps.append(pred_properties['P2z_exps'].cpu().detach().numpy())
        # test_correlation_xs.append(properties['correlation_xs'].numpy())
        # test_correlation_zs.append(properties['correlation_zs'].numpy())
        #
        # test_entanglement_entropies.append(pred_properties['entropies'].cpu().detach().numpy())
        # test_RMIs.append(properties['RMIs'].numpy())
        # test_inter_entropies.append(properties['inter_entropies'].numpy())

        # test_target_fidelites.append(properties['target_fidelities'].numpy())

        known_tgts = {}
        key_list = list(properties.keys())
        for ik in range(0, len(key_list)):
            tmp_key = key_list[ik]
            known_tgts[tmp_key] = properties[tmp_key].cuda(device=device_ids[0]).float()

        total_loss, key_loss = LossCriterion(pred_properties, known_tgts)
        all_key_loss = TestLossCriterion_ablation(pred_properties,known_tgts)
        all_test_key_losses.append(all_key_loss)

        for key in keys:
            key_losses[key] += key_loss[key]

        # if np.abs(properties['h1h2'].numpy()[0][0] - 0.12) < 0.01 and np.abs(
        #             properties['h1h2'].numpy()[0][1] - 0.73) < 0.01:
        #     print(properties['h1h2'].numpy())
        #     if total_loss < best_test_loss:
        #         pred_tgts = {}
        #         key_list = list(pred_properties.keys())
        #         for ik in range(0, len(key_list)):
        #             tmp_key = key_list[ik]
        #             pred_tgts[tmp_key] = pred_properties[tmp_key].cpu()
        #         with open('results/test_point'+str(properties['h1h2'].numpy()[0])+ '_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
        #             num_measure_qubits) + '_r_dim' + str(
        #             r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size300.pkl', 'wb') as f:
        #             pickle.dump(pred_tgts, f)
        #         with open('results/test_point'+str(properties['h1h2'].numpy()[0])+'_real_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
        #                 num_measure_qubits) + '_r_dim' + str(
        #             r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size300.pkl', 'wb') as f:
        #             pickle.dump(properties, f)
        #         print(key_loss)
        #         print(total_loss)
        #         best_test_loss = total_loss

        test_loss += total_loss


    print("train_loss:", end=" ")
    print(train_loss)
    # if train_loss < 7:
    #     optimizer.param_groups[0]['lr'] = 0.0001
    print(train_key_losses)
    print("-------------------------------------")
    print("test_loss:", end=" ")
    print(test_loss)
    print(key_losses)

    # if test_flag == 1 or (test_loss < best_loss and test_flag == 0):
    #     np.save('results/test_rs_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #         num_measure_qubits) + '_r_dim' + str(
    #         r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,np.array(test_rs))
    #     np.save('results/test_labels_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #         num_measure_qubits) + '_r_dim' + str(
    #         r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_labels)
    #     np.save('results/test_phase_values_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #         num_measure_qubits) + '_r_dim' + str(
    #         r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_phase_values)
    #     np.save('results/test_h1h2s_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #         num_measure_qubits) + '_r_dim' + str(
    #         r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_h1h2s)
    #
    #     # np.save('results/test_cxs_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #     #         test_correlation_xs)
    #     # np.save('results/test_czs_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #     #         test_correlation_zs)
    #     # np.save('results/test_Px_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #     #         test_Px_exps)
    #     # np.save('results/test_Pzxz_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #     #         test_Pzxz_exps)
    #     np.save('results/test_P2x_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #         num_measure_qubits) + '_r_dim' + str(
    #         r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #             test_P2x_exps)
    #     np.save('results/test_P2z_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #         num_measure_qubits) + '_r_dim' + str(
    #         r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #             test_P2z_exps)
    #
    #
    #     # np.save('results/test_entropies_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, test_entanglement_entropies)
    #     # np.save('results/test_RMIs_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #     #         test_RMIs)
    #     # np.save('results/test_inter_entropies_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size),
    #     #         test_inter_entropies)
    #
    #     # np.save('results/test_fidelities_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
    #     #     num_measure_qubits) + '_r_dim' + str(
    #     #     r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings,
    #     #         test_target_fidelites)

    if test_loss < best_loss and test_flag == 0:
        torch.save(model.state_dict(), 'models/ppmodel_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
            num_measure_qubits) + '_r_dim' + str(
            r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings)
        np.save('results/test_loss_num_qubits' + str(num_qubits) + 'num_measure_qubits' + str(
        num_measure_qubits) + '_r_dim' + str(
        r_dim) + '_num_known_mbases' + str(num_know_mbases) + '_train_size' + str(train_size)+key_settings, all_test_key_losses)
