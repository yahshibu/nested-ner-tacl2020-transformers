__author__ = 'max'
__maintainer__ = 'takashi'

from typing import List, Tuple, Dict
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter


class ChainCRF4NestedNER(nn.Module):
    def __init__(self, input_size: int, num_labels_i: int) -> None:
        """
        Args:
            input_size: int
                the dimension of the input.
            num_labels_i: int
                the number of inside labels of the crf layer
        """
        super(ChainCRF4NestedNER, self).__init__()
        self.input_size: int = input_size
        self.num_labels_i: int = num_labels_i * 4
        self.num_labels: int = self.num_labels_i + 3

        # state weight tensor
        self.state_nn: nn.Linear = nn.Linear(input_size, self.num_labels)
        self.trans_matrix: Tensor = Parameter(Tensor(self.num_labels, self.num_labels), False)

        indices_i, index_o, index_eos, index_bos = self.get_indices()
        indices_bs = []
        indices_is = []
        indices_es = []
        indices_ss = []
        for index_i in indices_i:
            indices_bs.append(index_i['B'])
            indices_is.append(index_i['I'])
            indices_es.append(index_i['E'])
            indices_ss.append(index_i['S'])
        self.register_buffer('indices_bs', torch.LongTensor(indices_bs))
        self.register_buffer('indices_is', torch.LongTensor(indices_is))
        self.register_buffer('indices_es', torch.LongTensor(indices_es))
        self.register_buffer('indices_ss', torch.LongTensor(indices_ss))
        self.index_o: int = index_o
        self.index_eos: int = index_eos
        self.index_bos: int = index_bos

        self.reset_parameters()

    def reset_parameters(self) -> None:
        negative_inf = -1e4
        nn.init.constant_(self.state_nn.weight, 0.)
        self.state_nn.bias.data[:self.index_eos] = 0.
        self.state_nn.bias.data[self.index_eos:] = negative_inf
        nn.init.constant_(self.trans_matrix, 0.)
        for i in self.indices_bs:
            self.trans_matrix.data[i, :i + 1] = negative_inf  # B-XXX ->
            self.trans_matrix.data[i, i + 3:] = negative_inf  # B-XXX ->
        for i in self.indices_is:
            self.trans_matrix.data[i, :i] = negative_inf  # I-XXX ->
            self.trans_matrix.data[i, i + 2:] = negative_inf  # I-XXX ->
        for i in self.indices_es:
            self.trans_matrix.data[i, self.indices_is] = negative_inf  # E-XXX -> I-YYY
            self.trans_matrix.data[i, self.indices_es] = negative_inf  # E-XXX -> E-YYY
            self.trans_matrix.data[i, self.index_bos] = negative_inf  # E-XXX -> BOS
        for i in self.indices_ss:
            self.trans_matrix.data[i, self.indices_is] = negative_inf  # S-XXX -> I-YYY
            self.trans_matrix.data[i, self.indices_es] = negative_inf  # S-XXX -> E-YYY
            self.trans_matrix.data[i, self.index_bos] = negative_inf  # S-XXX -> BOS
        self.trans_matrix.data[self.index_o, self.indices_is] = negative_inf  # O -> I-XXX
        self.trans_matrix.data[self.index_o, self.indices_es] = negative_inf  # O -> E-XXX
        self.trans_matrix.data[self.index_o, self.index_bos] = negative_inf  # O -> BOS
        self.trans_matrix.data[self.index_eos, :self.index_eos] = negative_inf  # EOS ->
        self.trans_matrix.data[self.index_eos, self.index_bos] = negative_inf  # EOS -> BOS
        self.trans_matrix.data[self.index_bos, self.indices_is] = negative_inf  # BOS -> I-XXX
        self.trans_matrix.data[self.index_bos, self.indices_es] = negative_inf  # BOS -> E-XXX
        self.trans_matrix.data[self.index_bos, self.index_bos] = negative_inf  # BOS -> BOS
        self.trans_matrix.requires_grad = False

    def get_indices(self) -> Tuple[List[Dict[str, int]], int, int, int]:
        indices_i = []
        for i in range(0, self.num_labels_i, 4):
            index_i = dict()
            index_i['B'] = i
            index_i['I'] = i + 1
            index_i['E'] = i + 2
            index_i['S'] = i + 3
            indices_i.append(index_i)
        index_o = self.num_labels_i
        index_eos = self.num_labels_i + 1
        index_bos = self.num_labels_i + 2
        return indices_i, index_o, index_eos, index_bos

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]

        """
        batch, length, _ = input.size()

        # compute out_s by tensor dot [batch, length, input_size] * [input_size, num_label]
        # thus out_s should be [batch, length, num_label] --> [batch, length, 1, num_label]
        out_s = self.state_nn(input)

        if mask is not None:
            out_s[:, :, self.index_eos] += (mask == 0).float() * 2e4

        # [batch, length, num_label, num_label]
        output = self.trans_matrix + out_s.unsqueeze(2)

        return output

    def loss(self, input: Tensor, target: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for negative log likelihood loss
        """
        batch, length, _ = input.size()
        energy = self.forward(input, mask=mask)
        # shape = [length, batch, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)

        # shape = [batch, num_label]
        partition = None

        # shape = [batch]
        batch_index = torch.arange(0, batch).type_as(input).long()
        prev_label = input.new_full((batch, ), self.index_bos).long()
        tgt_energy = input.new_zeros(batch)

        for t in range(length):
            # shape = [batch, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, self.index_bos, :]
            else:
                # shape = [batch, num_label]
                partition = torch.logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
            label = target_transpose[t]
            tgt_energy += curr_energy[batch_index, prev_label, label]
            prev_label = label

        return \
            torch.logsumexp(self.trans_matrix.data[:, self.index_eos].unsqueeze(0) + partition, dim=1) - tgt_energy, \
            energy

    def nests_loss(self, energy: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            energy: Tensor
                the energy tensor with shape = [length, num_label, num_label]
            target: Tensor
                the tensor of target labels with shape [length]

        Returns: Tensor
                A 0D tensor for negative log likelihood loss
        """
        length, _, _ = energy.size()

        num_label_3 = self.indices_is.size(0)

        indices_3 = torch.cat((self.indices_bs.unsqueeze(0),
                               self.indices_is.repeat((length - 2, 1)),
                               self.indices_es.unsqueeze(0)),
                              dim=0)

        # shape = [num_label]
        partition_1 = None
        partition_3 = None

        # shape = []
        prev_label = self.index_bos
        tgt_energy = 0

        for t in range(length):
            # shape = [num_label, num_label]
            curr_energy = energy[t]
            if t == 0:
                partition_1 = curr_energy[self.index_bos, :]
                partition_3 = energy.new_full((num_label_3, ), -1e4)
            else:
                # shape = [num_label]
                partition = partition_1.clone()
                partition[indices_3[t - 1]] = partition_3
                partition_1 = torch.logsumexp(curr_energy + partition_1.unsqueeze(1), dim=0)
                partition_3 = torch.logsumexp(curr_energy[:, indices_3[t]] + partition.unsqueeze(1), dim=0)
            label = target[t]
            tgt_energy += curr_energy[prev_label, label]
            prev_label = label

        t = length - 1
        curr_energy = self.trans_matrix.data[:, self.index_eos]
        partition = curr_energy + partition_1
        partition[indices_3[t]] = curr_energy[indices_3[t]] + partition_3
        return torch.logsumexp(partition, dim=0) - tgt_energy

    def decode(self, input: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            decoding results in shape [batch, length]
        """

        energy = self.forward(input, mask=mask)

        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, :self.index_bos, :self.index_bos]

        length, batch_size, num_label, _ = energy_transpose.size()

        batch_index = torch.arange(0, batch_size).type_as(input).long()
        pointer = batch_index.new_zeros((length, batch_size, num_label))
        back_pointer = batch_index.new_zeros((length, batch_size))

        pi = energy[:, 0, self.index_bos, :self.index_bos]
        pointer[0] = self.index_bos
        for t in range(1, length):
            pi, pointer[t] = torch.max(energy_transpose[t] + pi.unsqueeze(2), dim=1)
        pi = self.trans_matrix.data[:self.index_bos, self.index_eos].unsqueeze(0) + pi

        _, back_pointer[-1] = torch.max(pi, dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1), energy

    def decode_nest(self, energy: Tensor) -> Tensor:
        """
        Args:
            energy: Tensor
                the energy tensor with shape = [length, num_label, num_label]

        Returns: Tensor
            decoding nested results in shape [length]
        """

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy[:, :self.index_bos, :self.index_bos]

        length, num_label, _ = energy_transpose.size()

        num_label_3 = self.indices_is.size(0)

        indices_3 = torch.cat((self.indices_bs.unsqueeze(0),
                               self.indices_is.repeat((length - 2, 1)),
                               self.indices_es.unsqueeze(0)),
                              dim=0)

        pointer_1 = energy.new_zeros((length, num_label)).long()
        pointer_3 = energy.new_zeros((length, num_label)).long()
        back_pointer = pointer_3.new_zeros(length)

        pi_1 = energy[0, self.index_bos, :self.index_bos]
        pi_3 = energy.new_full((num_label_3, ), -1e4)
        pointer_1[0] = self.index_bos
        pointer_3[0] = self.index_bos
        for t in range(1, length):
            e_t = energy_transpose[t]
            pi = pi_1.clone()
            pi[indices_3[t - 1]] = pi_3
            pi_1, pointer_1[t] = torch.max(e_t + pi_1.unsqueeze(1), dim=0)
            pi_3, pointer_3[t, indices_3[t]] = torch.max(e_t[:, indices_3[t]] + pi.unsqueeze(1), dim=0)
        t = length - 1
        e_t = self.trans_matrix.data[:self.index_bos, self.index_eos]
        pi = e_t + pi_1
        pi[indices_3[t]] = e_t[indices_3[t]] + pi_3

        _, back_pointer[-1] = torch.max(pi, dim=0)
        t = length - 2
        while t > -1:
            if (indices_3[t + 1] == back_pointer[t + 1]).nonzero().numel() == 0:
                break
            pointer_last = pointer_3[t + 1]
            back_pointer[t] = pointer_last[back_pointer[t + 1]]
            t -= 1
        while t > -1:
            pointer_last = pointer_1[t + 1]
            back_pointer[t] = pointer_last[back_pointer[t + 1]]
            t -= 1

        return back_pointer
