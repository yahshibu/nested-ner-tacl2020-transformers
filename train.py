#!/usr/bin/env python
from typing import Dict
import sys
import os
import numpy as np
import pickle
from datetime import datetime
from random import shuffle
import torch
import torch.cuda
import torch.nn
import copy
import time

from config import config
from model.sequence_labeling import BiRecurrentConvCRF4NestedNER
from training.logger import get_logger
from training.utils import adjust_learning_rate, clip_model_grad, create_opt
from training.utils import pack_target, unpack_prediction
from util.evaluate import evaluate
from util.utils import Alphabet, load_dynamic_config


def get_f1(model: BiRecurrentConvCRF4NestedNER, mode: str, file_path: str = None) -> float:
    with torch.no_grad():
        model.eval()

        pred_all, pred, recall_all, recall = 0, 0, 0, 0
        gold_cross_num = 0
        pred_cross_num = 0
        if mode == 'dev':
            batch_zip = zip(dev_input_ids_batches,
                            dev_input_mask_batches,
                            dev_first_subtokens_batches,
                            dev_last_subtokens_batches,
                            dev_label_batches,
                            dev_mask_batches)
        elif mode == 'test':
            batch_zip = zip(test_input_ids_batches,
                            test_input_mask_batches,
                            test_first_subtokens_batches,
                            test_last_subtokens_batches,
                            test_label_batches,
                            test_mask_batches)
        else:
            raise ValueError

        f = None
        if file_path is not None:
            f = open(file_path, 'w')

        for input_ids_batch, input_mask_batch, first_subtokens_batch, last_subtokens_batch, label_batch, mask_batch \
                in batch_zip:
            input_ids_batch_var = torch.LongTensor(np.array(input_ids_batch))
            input_mask_batch_var = torch.LongTensor(np.array(input_mask_batch))
            mask_batch_var = torch.ByteTensor(np.array(mask_batch, dtype=np.uint8))
            if config.if_gpu:
                input_ids_batch_var = input_ids_batch_var.cuda()
                input_mask_batch_var = input_mask_batch_var.cuda()
                mask_batch_var = mask_batch_var.cuda()

            pred_sequence_entities = model.predict(input_ids_batch_var,
                                                   input_mask_batch_var,
                                                   first_subtokens_batch,
                                                   last_subtokens_batch,
                                                   mask_batch_var)
            pred_entities = unpack_prediction(model, pred_sequence_entities)
            p_a, p, r_a, r = evaluate(label_batch, pred_entities)

            gold_cross_num += 0
            pred_cross_num += 0

            pred_all += p_a
            pred += p
            recall_all += r_a
            recall += r

            if file_path is not None:
                for input_ids, input_mask, first_subtokens, last_subtokens, mask, label, preds \
                        in zip(input_ids_batch, input_mask_batch, first_subtokens_batch, last_subtokens_batch,
                               mask_batch, label_batch, pred_entities):
                    words = []
                    for t, m in zip(input_ids, input_mask):
                        if m == 0:
                            break
                        words.append(voc_dict.get_instance(t))
                    f.write(' '.join(words) + '\n')

                    labels = []
                    for l in sorted(label, key=lambda x: (x[0], x[1], x[2])):
                        s = first_subtokens[l[0]]
                        e = last_subtokens[l[1] - 1]
                        labels.append("{},{} {}".format(s, e, label_dict.get_instance(l[2])))
                    f.write('|'.join(labels) + '\n')

                    labels = []
                    for p in sorted(preds, key=lambda x: (x[0], x[1], x[2])):
                        s = first_subtokens[p[0]]
                        e = last_subtokens[p[1] - 1]
                        labels.append("{},{} {}".format(s, e, label_dict.get_instance(p[2])))
                    f.write('|'.join(labels) + '\n')

                    f.write('\n')

        if file_path is not None:
            f.close()

        pred = pred / pred_all if pred_all > 0 else 1.
        recall = recall / recall_all if recall_all > 0 else 1.
        f1 = 2 / ((1. / pred) + (1. / recall)) if pred > 0. and recall > 0. else 0.
        logger.info("{} precision: {:.2f}%, recall: {:.2f}%, F1: {:.2f}%"
                    .format(mode, pred * 100., recall * 100., f1 * 100.))
        # logger.info("Prediction Crossing: ", pred_cross_num)
        # logger.info("Gold Crossing: ", gold_cross_num)

        return f1


# prepare log file
serial_number = datetime.now().strftime('%y%m%d_%H%M%S')
log_file_path = config.model_path + "_" + serial_number + '.tmp'
if not os.path.isdir(config.model_root_path):
    os.makedirs(config.model_root_path, mode=0o755, exist_ok=True)
logger = get_logger('Nested Mention', file=log_file_path)


# load data
f = open(config.train_data_path, 'rb')
train_input_ids_batches, \
    train_input_mask_batches, \
    train_first_subtokens_batches, \
    train_last_subtokens_batches, \
    train_label_batches, \
    train_mask_batches \
    = pickle.load(f)
f.close()
f = open(config.dev_data_path, 'rb')
dev_input_ids_batches, \
    dev_input_mask_batches, \
    dev_first_subtokens_batches, \
    dev_last_subtokens_batches, \
    dev_label_batches, \
    dev_mask_batches \
    = pickle.load(f)
f.close()
f = open(config.test_data_path, 'rb')
test_input_ids_batches, \
    test_input_mask_batches, \
    test_first_subtokens_batches, \
    test_last_subtokens_batches, \
    test_label_batches, \
    test_mask_batches \
    = pickle.load(f)
f.close()

# misc info
misc_config: Dict[str, Alphabet] = pickle.load(open(config.config_data_path, 'rb'))
voc_dict, label_dict = load_dynamic_config(misc_config)
config.voc_size = voc_dict.size()
config.label_size = label_dict.size()

config.if_gpu = config.if_gpu and torch.cuda.is_available()

logger.info(config)  # print training setting

ner_model = BiRecurrentConvCRF4NestedNER(config.bert_model, config.label_size,
                                         hidden_size=config.hidden_size, layers=config.layers,
                                         lstm_dropout=config.lstm_dropout)
if config.if_gpu:
    ner_model = ner_model.cuda()

parameters = filter(lambda p: p.requires_grad, ner_model.parameters())
optimizer, lr_scheduler = create_opt(parameters, config.opt, lr=config.lr, l2=config.l2, lr_patience=config.lr_patience)

train_sequence_label_batches = [pack_target(ner_model, train_label_batch, train_mask_batch)
                                for train_label_batch, train_mask_batch in zip(train_label_batches, train_mask_batches)]

logger.info("{} batches expected for training".format(len(train_input_ids_batches)))
logger.info("")
best_model = None
best_per = float('-inf')
best_loss = float('inf')
train_all_batches = list(zip(train_input_ids_batches,
                             train_input_mask_batches,
                             train_first_subtokens_batches,
                             train_last_subtokens_batches,
                             train_sequence_label_batches,
                             train_mask_batches))

train_start_time = time.time()
num_batches = len(train_all_batches)
for e_ in range(1, config.epoch + 1):
    logger.info("Epoch {:d} (learning rate={:.4f}):".format(e_, optimizer.param_groups[0]['lr']))
    train_err = 0.
    train_total = 0.

    if config.if_shuffle:
        shuffle(train_all_batches)
    batch_counter = 0
    start_time = time.time()
    ner_model.train()
    num_back = 0
    for input_ids_batch, input_mask_batch, first_subtokens_batch, last_subtokens_batch, label_batch, mask_batch \
            in train_all_batches:
        batch_len = max([len(first_subtokens) for first_subtokens in first_subtokens_batch])

        input_ids_batch_var = torch.LongTensor(np.array(input_ids_batch))
        input_mask_batch_var = torch.LongTensor(np.array(input_mask_batch))
        mask_batch_var = torch.ByteTensor(np.array(mask_batch, dtype=np.uint8))
        if config.if_gpu:
            input_ids_batch_var = input_ids_batch_var.cuda()
            input_mask_batch_var = input_mask_batch_var.cuda()
            mask_batch_var = mask_batch_var.cuda()

        optimizer.zero_grad()
        loss = ner_model.forward(input_ids_batch_var, input_mask_batch_var, first_subtokens_batch, last_subtokens_batch,
                                 label_batch, mask_batch_var)
        loss.backward()
        clip_model_grad(ner_model, config.clip_norm)

        batch_counter += 1

        optimizer.step(None)

        with torch.no_grad():
            train_err += loss * batch_len
            train_total += batch_len

        # update log
        if batch_counter % 10 == 0:
            time_ave = (time.time() - start_time) / batch_counter
            time_left = (num_batches - batch_counter) * time_ave

            sys.stdout.write('\b' * num_back)
            sys.stdout.write(' ' * num_back)
            sys.stdout.write('\b' * num_back)
            log_info = "train: {:d}/{:d} loss: {:.4f}, time left (estimated): {:.2f}s" \
                       .format(batch_counter, num_batches, train_err / train_total, time_left)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write('\b' * num_back)
    sys.stdout.write(' ' * num_back)
    sys.stdout.write('\b' * num_back)
    logger.info("train: {:d} loss: {:.4f}, time: {:.2f}s"
                .format(num_batches, train_err / train_total, time.time() - start_time))

    if e_ % config.check_every != 0:
        continue

    # evaluating dev and always save the best
    cur_time = time.time()
    f1 = get_f1(ner_model, 'dev')
    logger.info("dev step took {:.4f} seconds".format(time.time() - cur_time))
    logger.info("")

    # early stop
    if f1 > best_per:
        best_per = f1
        del best_model
        best_model = copy.deepcopy(ner_model)
    if train_err < best_loss:
        best_loss = train_err
    if not adjust_learning_rate(lr_scheduler, e_, train_err, f1):
        break

logger.info("training step took {:.4f} seconds".format(time.time() - train_start_time))
logger.info("best dev F1: {:.2f}%".format(best_per * 100.))
logger.info("")

serial_number = datetime.now().strftime('%y%m%d_%H%M%S')
this_model_path = config.model_path + "_" + serial_number
if not os.path.isdir(config.model_root_path):
    os.makedirs(config.model_root_path, mode=0o755, exist_ok=True)

# remember to eval after loading the model. for the reason of batchnorm and dropout
cur_time = time.time()
f1 = get_f1(best_model, 'test', file_path=this_model_path + '.result.txt')
logger.info("test step took {:.4f} seconds".format(time.time() - cur_time))

logger.info("Dumping model to {}".format(this_model_path + '.pt'))
torch.save(best_model.state_dict(), this_model_path + '.pt')

os.rename(log_file_path, this_model_path + '.log.txt')
