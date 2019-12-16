__author__ = 'max'
__maintainer__ = 'takashi'

from typing import List, Union
import torch
from torch import Tensor
import torch.nn as nn
from transformers.modeling_bert import BertModel

from module.crf import ChainCRF4NestedNER
from module.dropout import VarDropout
from module.variational_rnn import VarMaskedFastLSTM


class NestedSequenceLabel:
    def __init__(self, start: int, end: int, label: Tensor, children: List) -> None:
        self.start = start
        self.end = end
        self.label = label
        self.children = children


class BiRecurrentConvCRF4NestedNER(nn.Module):
    def __init__(self, bert_model: str, label_size: int, hidden_size: int = 256, layers: int = 1,
                 lstm_dropout: float = 0.50, fine_tune: bool = False) -> None:
        super(BiRecurrentConvCRF4NestedNER, self).__init__()

        self.bert: BertModel = BertModel.from_pretrained(bert_model)
        self.bert.embeddings.dropout = VarDropout(self.bert.embeddings.dropout.p)
        for l in range(len(self.bert.encoder.layer)):
            self.bert.encoder.layer[l].attention.output.dropout \
                = VarDropout(self.bert.encoder.layer[l].attention.output.dropout.p)
            self.bert.encoder.layer[l].output.dropout \
                = VarDropout(self.bert.encoder.layer[l].output.dropout.p)
        self.fine_tune: bool = fine_tune
        if fine_tune:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        else:
            for name, parameter in self.bert.named_parameters():
                parameter.requires_grad = False
            self.bert.encoder.output_hidden_states = True
        # standard dropout
        self.dropout_out: nn.Dropout2d = nn.Dropout2d(p=lstm_dropout)

        if fine_tune:
            self.rnn: VarMaskedFastLSTM = VarMaskedFastLSTM(self.bert.config.hidden_size, hidden_size,
                                                            num_layers=layers, batch_first=True, bidirectional=True,
                                                            dropout=(lstm_dropout, lstm_dropout))
        else:
            self.bert_layers: int = 8
            self.rnn: VarMaskedFastLSTM = VarMaskedFastLSTM(self.bert.config.hidden_size * self.bert_layers,
                                                            hidden_size, num_layers=layers,
                                                            batch_first=True, bidirectional=True,
                                                            dropout=(lstm_dropout, lstm_dropout))

        self.reset_parameters()

        self.all_crfs: List[ChainCRF4NestedNER] = []

        for label in range(label_size):
            crf = ChainCRF4NestedNER(hidden_size * 2, 1)
            self.all_crfs.append(crf)
            self.add_module('crf%d' % label, crf)

        self.b_id: int = 0
        self.i_id: int = 1
        self.e_id: int = 2
        self.s_id: int = 3
        self.o_id: int = 4
        self.eos_id: int = 5

    def reset_parameters(self) -> None:
        for name, parameter in self.rnn.named_parameters():
            nn.init.constant_(parameter, 0.)
            if name.find('weight_ih') > 0:
                if name.startswith('cell0.weight_ih') or name.startswith('cell1.weight_ih'):
                    bound = (6. / (self.rnn.input_size + self.rnn.hidden_size)) ** 0.5
                else:
                    bound = (6. / ((2 * self.rnn.hidden_size) + self.rnn.hidden_size)) ** 0.5
                nn.init.uniform_(parameter, -bound, bound)
                parameter.data[:2, :, :] = 0.
                parameter.data[3:, :, :] = 0.
            if name.find('bias_hh') > 0:
                parameter.data[1, :] = 1.

    def _get_rnn_output(self, input_ids: Tensor, input_mask: Tensor,
                        first_subtokens: List[List[int]], last_subtokens: List[List[int]], mask: Tensor = None) \
            -> Tensor:
        # [batch, length, word_dim]
        with torch.set_grad_enabled(self.fine_tune and torch.is_grad_enabled()):
            sequence_output = self.bert(input_ids, attention_mask=input_mask)
            if self.fine_tune:
                sequence_output = sequence_output[0]
            else:
                sequence_output = torch.cat(tuple(sequence_output[2][-self.bert_layers:]), 2).detach()
            batch, _, word_dim = sequence_output.size()
            input = sequence_output.new_zeros((batch, max([len(fst) for fst in first_subtokens]), word_dim))
            for i, subtokens_list_tuple in enumerate(zip(first_subtokens, last_subtokens)):
                for j, subtokens_tuple in enumerate(zip(subtokens_list_tuple[0], subtokens_list_tuple[1])):
                    input[i, j, :] = torch.mean(sequence_output[i, subtokens_tuple[0]:subtokens_tuple[1], :], dim=0)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask)

        # apply dropout for the output of rnn
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output

    def forward(self, input_ids: Tensor, input_mask: Tensor,
                first_subtokens: List[List[int]], last_subtokens: List[List[int]],
                target: Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]], mask: Tensor) -> Tensor:
        # output from rnn [batch, length, tag_space]
        output = self._get_rnn_output(input_ids, input_mask, first_subtokens, last_subtokens, mask=mask)

        # [batch, length, num_label, num_label]
        batch, length, _ = output.size()

        loss = []

        for label, crf in enumerate(self.all_crfs):
            target_batch = torch.cat(tuple([target_each.label.unsqueeze(0) for target_each in target[label]]), dim=0)

            loss_batch, energy_batch = crf.loss(output, target_batch, mask=mask)

            calc_nests_loss = crf.nests_loss

            def forward_recursively(loss: Tensor, energy: Tensor, target: NestedSequenceLabel, offset: int) -> Tensor:
                nests_loss_list = []
                for child in target.children:
                    if child.end - child.start > 1:
                        nests_loss = calc_nests_loss(energy[child.start - offset:child.end - offset, :, :],
                                                     child.label)
                        nests_loss_list.append(forward_recursively(nests_loss,
                                                                   energy[child.start - offset:child.end - offset, :, :],
                                                                   child, child.start))
                return sum(nests_loss_list) + loss

            loss_each = []
            for i in range(batch):
                loss_each.append(forward_recursively(loss_batch[i], energy_batch[i], target[label][i], 0))

            loss.append(sum(loss_each))

        loss = sum(loss)

        return loss / batch

    def predict(self, input_ids: Tensor, input_mask: Tensor,
                first_subtokens: List[List[int]], last_subtokens: List[List[int]], mask: Tensor) \
            -> Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]]:
        # output from rnn [batch, length, tag_space]
        output = self._get_rnn_output(input_ids, input_mask, first_subtokens, last_subtokens, mask=mask)

        batch, length, _ = output.size()

        preds = []

        for crf in self.all_crfs:
            preds_batch, energy_batch = crf.decode(output, mask=mask)

            b_id = self.b_id
            i_id = self.i_id
            e_id = self.e_id
            o_id = self.o_id
            eos_id = self.eos_id
            decode_nest = crf.decode_nest

            def predict_recursively(preds: Tensor, energy: Tensor, offset: int) -> NestedSequenceLabel:
                length = preds.size(0)
                nested_preds_list = []
                index = 0
                while index < length:
                    id = preds[index]
                    if id == eos_id:
                        break
                    if id != o_id:
                        if id == b_id:  # B-XXX
                            start_tmp = index
                            index += 1
                            if index == length:
                                break
                            id = preds[index]
                            while id == i_id:  # I-XXX
                                index += 1
                                if index == length:
                                    break
                                id = preds[index]
                            if id == e_id:  # E-XXX
                                end_tmp = index + 1
                                nested_preds = decode_nest(energy[start_tmp:end_tmp, :, :])
                                nested_preds_list.append(predict_recursively(nested_preds,
                                                                             energy[start_tmp:end_tmp, :, :],
                                                                             start_tmp + offset))
                    index += 1
                return NestedSequenceLabel(offset, length + offset, preds, nested_preds_list)

            preds_each = []
            for i in range(batch):
                preds_each.append(predict_recursively(preds_batch[i, :], energy_batch[i, :, :, :], 0))

            preds.append(preds_each)

        return preds
