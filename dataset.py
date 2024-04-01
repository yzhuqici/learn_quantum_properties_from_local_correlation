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
class NoisyStateData(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3, shot_num=100, group=0):
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


        self.noisy_input_data = np.load(str(num_qubits)+'qubit_noisy/' + str(num_qubits) +'qubit_'+str(shot_num)+'shot_group'+str(group)+'_partial' + str(num_measure_qubits) + '_probs.npy')

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
        return self.observables, self.input_data[idx], self.noisy_input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)

class DNoiseStateData(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3,p=0.05):
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
        noisy_values = []
        for i in range(64, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            noisy_value = np.load(
                str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_dnoise_p' + str(p) + '_partial' + str(
                    num_measure_qubits) + '_probs' + str(i) + '.npy')
            noisy_value = noisy_value.reshape(-1, 2**num_measure_qubits)
            values.append(value)
            noisy_values.append(noisy_value)
        self.input_data = np.array(values)
        self.noisy_input_data = np.array(noisy_values)
        
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
        return self.observables, self.input_data[idx], self.noisy_input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)

class StateZZData(Dataset):
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
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_zz_probs'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        P2x_exps = np.load(
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values_zz.npy').real
        P2z_exps = np.load(
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values_zz.npy').real

        # phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels.npy').reshape(4096)
        phase_values = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_zz.npy').reshape(4096)

        entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies_zz.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(64, num_states):
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
class StateRandomGate(Dataset):
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
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector1_random_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector2_random_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        # h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        # P2x_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        # P2z_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels_random_gate.npy').reshape(num_states*2)
        phase_values1 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector1_random_gate.npy').reshape(num_states)
        phase_values2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector2_random_gate.npy').reshape(num_states)
        phase_values = np.concatenate((phase_values1,phase_values2),0)

        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states*2):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            # properties['P2x_exps'] = P2x_exps[i]
            # properties['P2z_exps'] = P2z_exps[i]
            #
            # properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            # properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)
class StateRandom2LayerGate(Dataset):
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
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector1_random_2layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector2_random_2layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        # h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        # P2x_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        # P2z_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels_random_gate.npy').reshape(num_states*2)
        phase_values1 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector1_random_2layer_gate.npy').reshape(num_states)
        phase_values2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector2_random_2layer_gate.npy').reshape(num_states)
        phase_values = np.concatenate((phase_values1,phase_values2),0)

        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states*2):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            # properties['P2x_exps'] = P2x_exps[i]
            # properties['P2z_exps'] = P2z_exps[i]
            #
            # properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            # properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)
class StateRandom4LayerGate(Dataset):
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
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector1_random_4layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector2_random_4layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        # h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        # P2x_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        # P2z_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels_random_gate.npy').reshape(num_states*2)
        phase_values1 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector1_random_4layer_gate.npy').reshape(num_states)
        phase_values2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector2_random_4layer_gate.npy').reshape(num_states)
        phase_values = np.concatenate((phase_values1,phase_values2),0)

        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states*2):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            # properties['P2x_exps'] = P2x_exps[i]
            # properties['P2z_exps'] = P2z_exps[i]
            #
            # properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            # properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)
class StateRandomLayerGate(Dataset):
    def __init__(self,num_states=30,num_layers=10, num_qubits=9, num_measure_qubits = 3):
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
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector1_random_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        for i in range(0, num_states):
            value = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector2_random_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        # h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        # P2x_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        # P2z_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels_random_gate.npy').reshape(num_states*2)
        phase_values1 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector1_random_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector2_random_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values = np.concatenate((phase_values1,phase_values2),0)

        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states*2):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            # properties['P2x_exps'] = P2x_exps[i]
            # properties['P2z_exps'] = P2z_exps[i]
            #
            # properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            # properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)
class StateRandomGGate(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3, num_layers=1):
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
        for i in range(0, num_states):
            value = np.load("randomG_"+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector1_randomG_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        for i in range(0, num_states):
            value = np.load("randomG_"+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector2_randomG_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        # h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        # P2x_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        # P2z_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels_random_gate.npy').reshape(num_states*2)
        phase_values1 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector1_randomG_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector2_randomG_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values = np.concatenate((phase_values1,phase_values2),0)

        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states*2):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            # properties['P2x_exps'] = P2x_exps[i]
            # properties['P2z_exps'] = P2z_exps[i]
            #
            # properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            # properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)
class StateRandomUGGate(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3, num_layers=1):
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
        for i in range(0, num_states):
            value = np.load("randomG_"+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector1_randomUG_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        for i in range(0, num_states):
            value = np.load("randomG_"+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector2_randomUG_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        # h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        # P2x_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        # P2z_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels_random_gate.npy').reshape(num_states*2)
        phase_values1 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector1_randomUG_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector2_randomUG_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values = np.concatenate((phase_values1,phase_values2),0)

        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states*2):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            # properties['P2x_exps'] = P2x_exps[i]
            # properties['P2z_exps'] = P2z_exps[i]
            #
            # properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            # properties['h1h2'] = h1h2[i]

            # properties['target_fidelities'] = target_fidelities[i]

            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)
class StateRandomGUGGate(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3, num_layers=1):
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
        for i in range(0, num_states):
            value = np.load("randomG_"+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector1_randomG_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        for i in range(0, num_states):
            value = np.load("randomG_"+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs_target_vector2_randomUG_'+str(num_layers)+'layer_gate_'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        # h1h2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(4096,-1)

        # correlation_xs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_correlation_xs_values.npy').real.reshape((4096,-1))
        # correlation_zs = np.load(str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_correlation_zs_values.npy').real.reshape((4096,-1))

        # Px_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulix_expct_values.npy').real.reshape((4096,-1))
        # Pzxz_exps = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_Paulizxz_expct_values.npy').real.reshape((4096,-1))

        # P2x_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        # P2z_exps = np.load(
        #     str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels_random_gate.npy').reshape(num_states*2)
        phase_values1 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector1_randomG_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values2 = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values_target_vector2_randomUG_'+str(num_layers)+'layer_gate.npy').reshape(num_states)
        phase_values = np.concatenate((phase_values1,phase_values2),0)

        # entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((4096,-1))
        # RMIs = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_RMI.npy').real.reshape((4096,-1))
        # inter_entropies = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_inter_entanglement.npy').real.reshape((4096,-1))
        # target_fidelities = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_target_fidelities.npy').real.reshape((4096,-1))

        for i in range(0, num_states*2):
            properties = {}

            # properties['Px_exps'] = Px_exps[i]
            # properties['Pzxz_exps'] = Pzxz_exps[i]
            # properties['correlation_xs'] = correlation_xs[i]
            # properties['correlation_zs'] = correlation_zs[i]
            # properties['P2x_exps'] = P2x_exps[i]
            # properties['P2z_exps'] = P2z_exps[i]
            #
            # properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]

            properties['phase_label'] = phase_labels[i]
            properties['phase_value'] = phase_values[i]
            # properties['h1h2'] = h1h2[i]

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
class NoisyHeisenbergData(Dataset):
    def __init__(self,num_states=30,num_qubits=50, num_measure_qubits = 3, shot_num=100, group=0):
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
        self.noisy_input_data = np.load('Heisenberg_noisy/prob3_'+str(shot_num)+'shot_group'+str(group)+ str(num_qubits) + 'qubits.npy')


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
        # if num_qubits == 50:
        indices = list(range(1, 64, 3))
        ZR_values = np.load('Heisenberg/'+str(num_qubits)+'_ZR_values.npy').reshape(64, 64)
        phase_values = ZR_values[indices][:, indices].reshape(-1)

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
            properties['phase_value'] = phase_values[i]
            properties['jpdelta'] = jpdeltas[i]
            #
            # properties['target_fidelities'] = target_fidelities[i]
            # properties['fidelities'] = fidelities[i]
            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.noisy_input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)

class HeisenbergData2(Dataset):
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
        # P2x_exps -= np.mean(P2x_exps,axis = 0)
        # P2x_exps /= np.std(P2x_exps, axis=0)
        P2x_exps2 = np.squeeze(np.load("Heisenberg/two_point2_xs_"+str(num_qubits) + 'qubits.npy'))
        P2x_exps = np.concatenate((P2x_exps,P2x_exps2),axis=1)

        P2z_exps = np.squeeze(np.load("Heisenberg/two_point_zs_"+str(num_qubits) + 'qubits.npy'))
        # P2z_exps -= np.mean(P2z_exps, axis=0)
        # P2z_exps /= np.std(P2z_exps, axis=0)
        P2z_exps2 = np.squeeze(np.load("Heisenberg/two_point2_zs_"+str(num_qubits) + 'qubits.npy'))
        P2z_exps = np.concatenate((P2z_exps, P2z_exps2), axis=1)

        RMIs = np.squeeze(np.load("Heisenberg/RMIs_"+str(num_qubits) + 'qubits.npy'))[:,0]
        # RMIs -= np.mean(RMIs, axis=0)
        # RMIs /= np.std(RMIs, axis=0)

        RMI2s = np.squeeze(np.load("Heisenberg/RMI2s_" + str(num_qubits) + 'qubits.npy'))
        # RMI2s -= np.mean(RMI2s, axis=0)
        # RMI2s /= np.std(RMI2s, axis=0)

        # fidelities = np.squeeze(np.load("Heisenberg/fidelities_"+str(num_qubits) + 'qubits.npy'))

        # phase_labels = np.load(str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels.npy').reshape(4096)
        indices = list(range(1, 64, 3))
        ZR_values = np.load("Heisenberg/ZR_values.npy").reshape(64, 64)
        phase_values = ZR_values[indices][:, indices].reshape(-1)

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
            properties['phase_value'] = phase_values[i]
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
class TwoD_Data(Dataset):
    def __init__(self,num_states=100,num_qubits=12, num_measure_qubits = 2):
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(
                str(num_measure_qubits) + 'qubit/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load('2D/' + str(num_qubits) + 'qubits_' + str(num_measure_qubits) +'measured_combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        for i in range(num_states):
            value = np.load('2D/' + str(num_qubits) + 'qubits_' + str(num_measure_qubits) +'measured_probs'+str(i)+'.npy')
            value = value.reshape(-1, 2 ** num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []

        ham_coeffs = []
        entropies = []
        for i in range(num_states):
            ham_coeff = np.load("2D/state_coeff_" + str(num_qubits) + 'qubits_' + str(i)+'.npy')
            ham_coeffs.append(ham_coeff)
            entropy = np.load('2D/state_' + str(num_qubits) + 'qubits_two_site_entanglement_entropies_'+str(i)+'.npy').real
            entropies.append(entropy)
        ham_coeffs = np.array(ham_coeffs)
        expectations = np.load('2D/' + str(num_qubits) +'qubit_two_point_expct_values.npy')

        entropies = np.array(entropies)

        for i in range(0, num_states):
            properties = {}

            properties['hs'] = ham_coeffs[i]
            # properties['P2z_exps'] = P2z_exps[i]
            properties['expectations'] = expectations[i]
            # properties['correlation_zs'] = correlation_zs[i]
            #
            properties['entropies'] = entropies[i]
            # properties['RMIs'] = RMIs[i]
            # properties['inter_entropies'] = inter_entropies[i]
            #
            # properties['phase_label'] = phase_labels[i]
            # properties['phase_value'] = phase_values[i]
            # properties['jpdelta'] = jpdeltas[i]
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
class cat_Data(Dataset):
    def __init__(self,num_states=2500):
        self.observables = np.load('cstate/phis_p300.npy')
        self.input_data = np.load('cstate/cat_probs.npy')
        mean_photon_nums = np.load("cstate/mean_photon_nums.npy")
        parities = np.load('cstate/parities.npy')
        nprobs = np.load('cstate/nprobs.npy')
        lw_funcs = np.load('cstate/lw_funcs.npy')
        output_properties = []
        for i in range(0, num_states):
            properties = {}

            properties['mpn'] = mean_photon_nums[i]/9
            properties['parity'] = parities[i].real
            properties['nprob'] = nprobs[i]
            properties['lw_func'] = lw_funcs[i]

            output_properties.append(properties)
        self.output_properties = output_properties

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]

    def __len__(self):
        return len(self.input_data)