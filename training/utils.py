from typing import Union, Iterator, Tuple, List
from enum import Enum
import numpy as np
import torch.nn
import torch.optim as optim
from adabound import AdaBound

from model.sequence_labeling import NestedSequenceLabel, BiRecurrentConvCRF4NestedNER


class Optimizer(Enum):
    AdaBound = 'AdaBound'
    SGD = 'SGD'
    Adam = 'Adam'


def adjust_learning_rate(lr_scheduler: Union[optim.lr_scheduler.StepLR, optim.lr_scheduler.ReduceLROnPlateau],
                         epoch: int, train_loss: float, dev_f1: float) -> bool:
    if isinstance(lr_scheduler, optim.lr_scheduler.StepLR):
        if isinstance(lr_scheduler.optimizer, AdaBound):
            lr_scheduler.step(epoch=epoch)
            return epoch < 200
        else:
            raise ValueError
    elif isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if isinstance(lr_scheduler.optimizer, optim.SGD):
            lr_scheduler.step(train_loss)
            return lr_scheduler.optimizer.param_groups[0]['lr'] >= 0.0001
        elif isinstance(lr_scheduler.optimizer, optim.Adam):
            lr_scheduler.step(dev_f1)
            return lr_scheduler.optimizer.param_groups[0]['lr'] >= 0.00001
        else:
            raise ValueError
    else:
        raise ValueError


def create_opt(parameters: Iterator, opt: Optimizer, lr: float = None, l2: float = None, lr_patience: int = None):
    if opt == Optimizer.AdaBound:
        optimizer = AdaBound(parameters, lr=lr if lr is not None else 0.001,
                             weight_decay=l2 if l2 is not None else 0.)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 150, gamma=0.1)
    elif opt == Optimizer.SGD:
        optimizer = optim.SGD(parameters, lr=lr if lr is not None else 0.1,
                              weight_decay=l2 if l2 is not None else 0.)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                            patience=lr_patience if lr_patience is not None else 5)
    elif opt == Optimizer.Adam:
        optimizer = optim.Adam(parameters, lr=lr if lr is not None else 0.001,
                               weight_decay=l2 if l2 is not None else 0.)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                            patience=lr_patience if lr_patience is not None else 3)
    else:
        raise ValueError
    return optimizer, lr_scheduler


def clip_model_grad(model: torch.nn.Module, clip_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type=2)


def pack_target(model: BiRecurrentConvCRF4NestedNER,
                flat_region_label_batch: List[List[Tuple[int, int, int]]], mask_batch: List[List[bool]]) \
        -> Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]]:

    def flat2nested(flat_label_list: List[Tuple[int, int, int]],
                    nested_label_list: List[Tuple[int, int, int, List]],
                    start: int, end: int, index: int, label: int) -> int:
        while index < len(flat_label_list):
            flat_label = flat_label_list[index]
            if flat_label[2] != label:
                index += 1
                continue
            if end <= flat_label[0]:
                break
            elif start <= flat_label[0] and flat_label[1] <= end:
                index += 1
                nested_nested_label_list = []
                index = flat2nested(flat_label_list, nested_nested_label_list, flat_label[0], flat_label[1], index,
                                    label)
                nested_label_list.append((flat_label[0], flat_label[1], flat_label[2], nested_nested_label_list))
            else:
                index += 1
                continue
        return index

    b_id = model.b_id
    i_id = model.i_id
    e_id = model.e_id
    s_id = model.s_id
    o_id = model.o_id
    eos_id = model.eos_id

    def region2sequence(region_label_list: List[Tuple[int, int, int, List]], start: int, end: int,
                        mask: List[bool] = None) -> NestedSequenceLabel:
        sequence_label = [o_id] * end
        if mask is not None and not mask[-1]:
            length = mask.index(False)
            sequence_label[length:] = [eos_id] * (end - length)
        nested_sequence_label_list = []
        for region_label in region_label_list:
            if region_label[1] - region_label[0] == 1:
                sequence_label[region_label[0]] = s_id  # S-XXX
                nested_sequence_label_list.append(
                    region2sequence(region_label[3], region_label[0], region_label[1]))
            else:
                sequence_label[region_label[0]] = b_id  # B-XXX
                sequence_label[region_label[0] + 1:region_label[1] - 1] \
                    = [i_id] * (region_label[1] - region_label[0] - 2)  # I-XXX
                sequence_label[region_label[1] - 1] = e_id  # E-XXX
                nested_sequence_label_list.append(
                    region2sequence(region_label[3], region_label[0], region_label[1]))
        sequence_label = torch.LongTensor(np.array(sequence_label[start:]))
        return NestedSequenceLabel(start, end, sequence_label, nested_sequence_label_list)

    nested_sequence_label_batch = []
    for flat_region_label_list, mask in zip(flat_region_label_batch, mask_batch):
        flat_region_label_list.sort(key=lambda x: (x[0], -x[1]))
        nested_sequence_label_batch_each = []
        for label in range(len(model.all_crfs)):
            nested_region_label_list = []
            flat2nested(flat_region_label_list, nested_region_label_list, 0, len(mask), 0, label)
            nested_sequence_label_batch_each.append(region2sequence(nested_region_label_list, 0, len(mask), mask))
        nested_sequence_label_batch.append(nested_sequence_label_batch_each)
    return list(map(list, zip(*nested_sequence_label_batch)))


def unpack_prediction(model: BiRecurrentConvCRF4NestedNER,
                      nested_sequence_label_batch: Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]]) \
        -> List[List[Tuple[int, int, int]]]:

    b_id = model.b_id
    i_id = model.i_id
    e_id = model.e_id
    s_id = model.s_id
    o_id = model.o_id
    eos_id = model.eos_id

    def sequence2region(sequence_label_tuple: NestedSequenceLabel, label: int) -> List[Tuple[int, int, int, List]]:
        start = sequence_label_tuple.start
        sequence_label = sequence_label_tuple.label.cpu().numpy()
        nested_region_label_list = []
        index = 0
        while index < len(sequence_label):
            start_tmp = None
            end_tmp = None
            flag = False
            label_tmp = sequence_label[index]
            if label_tmp == eos_id:
                break
            if label_tmp != o_id:
                if label_tmp == s_id:  # S-XXX
                    start_tmp = start + index
                    end_tmp = start + index + 1
                    flag = True
                elif label_tmp == b_id:  # B-XXX
                    start_tmp = start + index
                    index += 1
                    if index == len(sequence_label):
                        break
                    label_tmp = sequence_label[index]
                    while label_tmp == i_id:  # I-XXX
                        index += 1
                        if index == len(sequence_label):
                            break
                        label_tmp = sequence_label[index]
                    if label_tmp == e_id:  # E-XXX
                        end_tmp = start + index + 1
                        flag = True
            if flag:
                nested_sequence_tuple = None
                for nested_sequence_tuple_tmp in sequence_label_tuple.children:
                    if nested_sequence_tuple_tmp.start == start_tmp \
                            and nested_sequence_tuple_tmp.end == end_tmp:
                        nested_sequence_tuple = nested_sequence_tuple_tmp
                        break
                if nested_sequence_tuple is not None:
                    nested_region_label_list.append((start_tmp, end_tmp, label,
                                                     sequence2region(nested_sequence_tuple, label)))
                else:
                    nested_region_label_list.append((start_tmp, end_tmp, label, []))
            index += 1
        return nested_region_label_list

    def nested2flat(nested_label_list: List[Tuple[int, int, int, List]],
                    flat_label_list: List[Tuple[int, int, int]]) -> None:
        for nested_label in nested_label_list:
            flat_label_list.append((nested_label[0], nested_label[1], nested_label[2]))
            nested2flat(nested_label[3], flat_label_list)

    flat_region_label_batch = []
    for nested_sequence_label_tuple in list(map(list, zip(*nested_sequence_label_batch))):
        nested_region_label_list = []
        for label in range(len(model.all_crfs)):
            nested_region_label_list.extend(sequence2region(nested_sequence_label_tuple[label], label))
        flat_region_label_list = []
        nested2flat(nested_region_label_list, flat_region_label_list)
        flat_region_label_batch.append(flat_region_label_list)
    return flat_region_label_batch
