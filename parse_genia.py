#!/usr/bin/env python
from typing import Dict, Optional, Tuple, List
import os

from config import config


# Download http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Part-of-speech/GENIAcorpus3.02p.tgz
CORPUS_FILE_PATH: str = "../GENIA/GENIAcorpus3.02.merged.xml"


class Stat:

    def __init__(self) -> None:
        self.total: int = 0
        self.layer: List[int] = []
        self.ignored: int = 0
        self.num_labels: int = 0


TAG_SET: Dict[str, Stat] = {'G#DNA': Stat(),
                            'G#RNA': Stat(),
                            'G#protein': Stat(),
                            'G#cell_line': Stat(),
                            'G#cell_type': Stat()}


class Label:

    def __init__(self) -> None:
        self.start: Optional[int] = None
        self.end: Optional[int] = None
        self.tag: Optional[str] = None

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.end == other.end and self.tag == other.tag

    def __str__(self) -> str:
        return str(self.start) + ',' + str(self.end) + ' ' + self.tag


def calc_stat(words: List[str], labels: List[Label]) -> None:
    labels = sorted(labels, key=lambda x: (x.start, -x.end, x.tag))
    for tag, stat in TAG_SET.items():
        sequence_label = [0] * len(words)
        prev_label = None
        for label in labels:

            if label.tag != tag:
                continue
            stat.total += 1

            if prev_label is not None and label == prev_label:
                depth = sequence_label[label.start] - 1
                stat.layer[depth] += 1
                continue

            flag = True
            depth = sequence_label[label.start]
            for index in range(label.start + 1, label.end):
                if sequence_label[index] != depth:
                    flag = False
                    break

            if flag:
                for index in range(label.start, label.end):
                    sequence_label[index] += 1
                if len(stat.layer) == depth:
                    stat.layer.append(0)
                stat.layer[depth] += 1
            else:
                stat.ignored += 1

            prev_label = label

        stat.num_labels += sum(sequence_label) + len(words)


SENTENCE_BEGIN_TAG = '<sentence>'
SENTENCE_END_TAG = '</sentence>'
MENTION_BEGIN_TAG = '<cons '
MENTION_END_TAG = '</cons>'
WORD_BEGIN_TAG = '<w '
WORD_END_TAG = '</w>'

LEX_ATTRIBUTE = ' lex="'
SEM_ATTRIBUTE = ' sem="'


def parse_line(line: str, do_lower_case: bool) -> Tuple[str, str]:

    if line.find('HMG-I(Y)</cons>') > -1:
        line = line.replace('HMG-I(Y)</cons>', '<w c="NN">HMG-I(Y)</w></cons>')

    # words
    word_tags_begin = []
    words_begin = []
    index = -1
    while True:
        index = line.find(WORD_BEGIN_TAG, index + 1)
        if index < 0:
            break
        word_tags_begin.append(index)
        index = line.find('>', index + 1)
        words_begin.append(index + 1)

    words_end = []
    index = -1
    while True:
        index = line.find(WORD_END_TAG, index + 1)
        if index < 0:
            break
        words_end.append(index)

    assert (len(words_begin) == len(words_end))
    words = list()
    for bi, ei in zip(words_begin, words_end):
        word = line[bi:ei]
        if do_lower_case:
            word = word.lower()
        words.append(word.replace(' ', '\xa0'))

    # labels
    mention_tags_begin = []
    mentions_begin = []
    index = -1
    while True:
        index = line.find(MENTION_BEGIN_TAG, index + 1)
        if index < 0:
            break
        mention_tags_begin.append(index)
        index = line.find('>', index + 1)
        mentions_begin.append(index)

    mentions_end = []
    index = -1
    while True:
        index = line.find(MENTION_END_TAG, index + 1)
        if index < 0:
            break
        mentions_end.append(index)

    assert (len(mentions_begin) == len(mentions_end))
    tags = []
    for bi, ei in zip(mention_tags_begin, mentions_begin):
        bi2 = line.find(SEM_ATTRIBUTE, bi, ei)
        if bi2 < 0:
            tags.append(None)
            continue
        bi2 += len(SEM_ATTRIBUTE)
        ei2 = line.index('"', bi2, ei)
        tags.append(line[bi2:ei2])

    stack = []
    que = []
    for index in range(len(line)):
        if index in mentions_begin:
            label = Label()
            label.start = len([i for i in words_begin if i < index])
            label.tag = tags.pop(0)
            stack.append(label)
        elif index in mentions_end:
            label = stack.pop()
            label.end = len([i for i in words_end if i <= index])
            que.append(label)

    labels = list()
    for label in que:
        for tag in TAG_SET:
            if label.tag is not None and label.tag.find(tag) > -1:
                label.tag = tag
                labels.append(label)
                break

    calc_stat(words, labels)

    return ' '.join(words), '|'.join([str(label) for label in labels])


def parse_genia() -> None:

    output_dir_path = "data/genia/"
    os.makedirs(output_dir_path, mode=0o755, exist_ok=True)

    output_file_list = ["genia.train", "genia.dev", "genia.test"]
    dataset_size_list = [15022, 1669, 1855]

    do_lower_case = '-cased' not in config.bert_model
    with open(CORPUS_FILE_PATH, 'r') as f:
        for output_file, dataset_size in zip(output_file_list, dataset_size_list):
            output_lines = []
            sent_count = 0
            token_count = 0
            for tag in TAG_SET:
                TAG_SET[tag] = Stat()
            for line in f:
                line = line.strip()
                if line.find(SENTENCE_BEGIN_TAG) > -1:
                    assert (line.find(SENTENCE_END_TAG) > -1)
                    words, labels = parse_line(line, do_lower_case)
                    output_lines.append(words + '\n')
                    output_lines.append(labels + '\n')
                    output_lines.append('\n')
                    sent_count += 1
                    token_count += len(words.split(' '))

                    if sent_count == dataset_size:
                        with open(output_dir_path + output_file, 'w') as f2:
                            f2.writelines(output_lines)

                        print("")
                        print("--- {}".format(output_file))
                        print("# of sentences:\t{:6d}".format(sent_count))
                        print("# of tokens:\t{:6d}".format(token_count))
                        total = 0
                        total_layer = []
                        total_ignored = 0
                        for _, stat in TAG_SET.items():
                            total += stat.total
                            for depth, num in enumerate(stat.layer):
                                if len(total_layer) == depth:
                                    total_layer.append(0)
                                total_layer[depth] += num
                            total_ignored += stat.ignored
                        print("total # of mentions:\t{}\t(layer:\t{},\tignored:\t{})"
                              .format(total, total_layer, total_ignored))
                        for tag, stat in TAG_SET.items():
                            print("\t{}:\t{:5d}\t(layer:\t{},\tignored:\t{})"
                                  .format(tag, stat.total, stat.layer, stat.ignored))
                        ave_labels = 0
                        for _, stat in TAG_SET.items():
                            ave_labels += stat.num_labels
                        ave_labels /= token_count * len(TAG_SET)
                        print("average # of labels:\t{:.2f}".format(ave_labels))

                        break


parse_genia()
