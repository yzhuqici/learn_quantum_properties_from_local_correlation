import torch
from torch.utils.data import Dataset
import numpy as np

class StateData(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3):
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(
                str(num_measure_qubits) + 'qubit/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_' + str(num_measure_qubits) +'combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        for i in range(64, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        P2x_exps = np.load(
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        P2z_exps = np.load(
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels.npy').reshape(4096)
        phase_values = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values.npy').reshape(4096)

        entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(64, num_states):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            properties['correlation_xs'] = correlation_xs[i]
            properties['correlation_zs'] = correlation_zs[i]
            properties['P2x_exps'] = P2x_exps[i]
            properties['P2z_exps'] = P2z_exps[i]

            properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)

class StateXXZZData(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3):
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(
                str(num_measure_qubits) + 'qubit/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_' + str(num_measure_qubits) +'combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        for i in range(64*16, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_xx_zz_probs'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2h3s.npy').reshape(64*64*16,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        P2x_exps = np.load(
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values_xx_zz.npy').real
        P2z_exps = np.load(
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values_xx_zz.npy').real

        # phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels.npy').reshape(4096)
        phase_values = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_xx_zz.npy').reshape(64*64*16)

        entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies_xx_zz.npy').real.reshape((64*64*16,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(64*16, num_states):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            properties['P2x_exps'] = P2x_exps[i]
            properties['P2z_exps'] = P2z_exps[i]

            properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            # properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)

class HeisenbergData(Dataset):
    def __init__(self,num_states=30,num_qubits=50, num_measure_qubits = 3):
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(
                str(num_measure_qubits) + 'qubit/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_' + str(num_measure_qubits) +'combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        J_ps = np.linspace(0, 3, 64)
        deltas = np.linspace(0, 4, 64)
        for id in range(1, len(deltas), 3):
            for iJ in range(1, len(J_ps), 3):
                J_p = J_ps[iJ]
                delta = deltas[id]
                value = np.load('Heisenberg/prob3_' + str(num_qubits) + 'qubits_Jp' + str(J_p) + '_delta' + str(delta)+'.npy')
                value = value.reshape(-1, 2 ** num_measure_qubits)
                values.append(value)
        self.input_data = np.array(values)


        output_properties = []

        P2x_exps = np.squeeze(np.load("Heisenberg/two_point_xs_"+str(num_qubits) + 'qubits.npy'))
        # P2x_exps2 = np.squeeze(np.load("Heisenberg/two_point2_xs_"+str(num_qubits) + 'qubits.npy'))
        # P2x_exps = np.concatenate((P2x_exps,P2x_exps2),axis=1)

        P2z_exps = np.squeeze(np.load("Heisenberg/two_point_zs_"+str(num_qubits) + 'qubits.npy'))
        # P2z_exps2 = np.squeeze(np.load("Heisenberg/two_point2_zs_"+str(num_qubits) + 'qubits.npy'))
        # P2z_exps = np.concatenate((P2z_exps, P2z_exps2), axis=1)
        if num_qubits == 50:
            RMIs = np.squeeze(np.load("Heisenberg/RMIs_"+str(num_qubits) + 'qubits.npy'))[:,0]
        else:
            RMIs = np.squeeze(np.load("Heisenberg/RMIs_" + str(num_qubits) + 'qubits.npy'))
        RMI2s = np.squeeze(np.load("Heisenberg/RMI2s_" + str(num_qubits) + 'qubits.npy'))

        # fidelities = np.squeeze(np.load("Heisenberg/fidelities_"+str(num_qubits) + 'qubits.npy'))

        # phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels.npy').reshape(4096)
        if num_qubits == 50:
            indices = list(range(1, 64, 3))
            ZR_values = np.load('Heisenberg/'+str(num_qubits)+'_ZR_values.npy').reshape(64, 64)
            phase_values = ZR_values[indices][:, indices].reshape(-1)
        elif num_qubits == 10:
            ZR_values = np.load('Heisenberg/' + str(num_qubits) + '_ZR_values.npy').reshape(21, 21)
            phase_values = ZR_values.reshape(-1)
        else:
            pass
        J_ps = np.linspace(0, 3, 64)
        deltas = np.linspace(0, 4, 64)
        jpdeltas = []
        for id in range(1, len(deltas), 3):
            for iJ in range(1, len(J_ps), 3):
                J_p = J_ps[iJ]
                delta = deltas[id]
                jpdeltas.append([J_p,delta])
        jpdeltas = np.array(jpdeltas)

        #
        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states):
            properties = {}

            properties['P2x_exps'] = P2x_exps[i]
            properties['P2z_exps'] = P2z_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            #
            # properties['entropies'] = entropies[i]
            properties['RMIs'] = RMIs[i]
            properties['RMI2s'] = RMI2s[i]
            # properties['inter_entropies'] = inter_entropies[i]
            #
            # properties['phase_label'] = phase_labels[i]
            try:
                properties['phase_value'] = phase_values[i]
            except:
                pass
            properties['jpdelta'] = jpdeltas[i]
            #
            # properties['target_fidelities'] = target_fidelities[i]
            # properties['fidelities'] = fidelities[i]
            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)

