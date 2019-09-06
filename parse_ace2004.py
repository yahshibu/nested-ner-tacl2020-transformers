#!/usr/bin/env python
from typing import List, Dict
import os
import re
from stanfordnlp.server import CoreNLPClient
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer

from config import config


# Download http://www.statnlp.org/research/ie/code/statnlp-mentionextraction.v0.2.tgz
SPLIT_INFO_DIR_PATH: str = "../statnlp-mentionextraction.v0.2/data/ACE2004_split/"
SPLIT_INFO_FILE_LIST: List[str] = ["train.txt", "dev.txt", "test.txt"]

# Get ACE 2004 Multilingual Training Corpus ( https://catalog.ldc.upenn.edu/LDC2005T09 )
CORPUS_DIR_PATH: str = "../ACE2004/ace_tides_multling_train/data/English/"


class Stat:

    def __init__(self) -> None:
        self.total: int = 0
        self.layer: List[int] = []
        self.ignored: int = 0
        self.num_labels: int = 0


TAG_SET: Dict[str, Stat] = {'FAC': Stat(),
                            'GPE': Stat(),
                            'LOC': Stat(),
                            'ORG': Stat(),
                            'PER': Stat(),
                            'VEH': Stat(),
                            'WEA': Stat()}


class EntityAnnotation:

    def __init__(self, start: int, end: int, type: str, mention: str) -> None:
        self.start: int = start
        self.end: int = end
        self.type: str = type
        self.mention: str = mention


class Token:

    def __init__(self, word: str, begin: int, end: int) -> None:
        self.word: str = word
        self.begin: int = begin
        self.end: int = end


class Tokenizer:

    def __init__(self) -> None:
        os.environ['CORENLP_HOME'] = '{}/stanford-corenlp-full-2018-10-05'.format(os.environ['HOME'])
        self.client = CoreNLPClient(annotators=['ssplit'])
        self.client.ensure_alive()
        self.do_lower_case = '-cased' not in config.bert_model
        self.basic_tokenizer: BasicTokenizer \
            = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=self.do_lower_case).basic_tokenizer

    def tokenize(self, doc: str) -> List[List[Token]]:
        corenlp_annotation = self.client.annotate(doc)
        sentences = []
        for sentence in corenlp_annotation.sentence:
            text = doc[sentence.characterOffsetBegin:sentence.characterOffsetEnd]
            if self.do_lower_case:
                text = text.lower()
            offset = sentence.characterOffsetBegin
            bert_tokens = self.basic_tokenizer.tokenize(text)
            begin = 0
            tokens = []
            for bert_token in bert_tokens:
                word = bert_token
                begin = text.index(word, begin)
                end = begin + len(word)
                tokens.append(Token(word, begin + offset, end + offset))
                begin = end
            if len(tokens) > 0:
                sentences.append(tokens)
        return sentences


class Label:

    def __init__(self) -> None:
        self.start: int = None
        self.end: int = None
        self.tag: str = None

    def __repr__(self) -> str:
        return str(self.start) + ',' + str(self.end) + ' ' + self.tag


def calc_stat(words: List[str], labels: List[Label]) -> None:
    labels = sorted(labels, key=lambda x: (x.start, -x.end, x.tag))
    for tag, stat in TAG_SET.items():
        sequence_label = [0] * len(words)
        for id, label in enumerate(labels):

            if label.tag != tag:
                continue
            stat.total += 1

            if id > 0 and label.__repr__() == labels[id - 1].__repr__():
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

        stat.num_labels += sum(sequence_label) + len(words)


ENTITY_BEGIN_TAG = '<entity '
ENTITY_END_TAG = '</entity>'
EXTENT_BEGIN_TAG = '<extent>'
EXTENT_END_TAG = '</extent>'
CHARSEQ_BEGIN_TAG = '<charseq '
CHARSEQ_END_TAG = '</charseq>'

TYPE_ATTRIBUTE = ' TYPE="'
START_ATTRIBUTE = ' START="'
END_ATTRIBUTE = ' END="'

TEXT_BEGIN_TAG = '<TEXT>'
TEXT_END_TAG = '</TEXT>'


def parse_document(basename: str, tokenizer: Tokenizer) -> List[str]:

    entity_annotations = []
    with open(basename + '.apf.xml', 'r') as f:
        entity_flag = False
        extent_flag = False
        tag = None
        while True:
            line = f.readline()
            if not line:
                break

            if line.find(ENTITY_BEGIN_TAG) > -1:
                entity_flag = True
                bi = line.index(TYPE_ATTRIBUTE) + len(TYPE_ATTRIBUTE)
                ei = line.index('"', bi)
                tag = line[bi:ei]

            if entity_flag:
                if line.find(EXTENT_BEGIN_TAG) > -1:
                    extent_flag = True

                if extent_flag:
                    if line.find(CHARSEQ_BEGIN_TAG) > -1:
                        bi = line.index(START_ATTRIBUTE) + len(START_ATTRIBUTE)
                        ei = line.index('"', bi)
                        start = int(line[bi:ei])
                        bi = line.index(END_ATTRIBUTE) + len(END_ATTRIBUTE)
                        ei = line.index('"', bi)
                        end = int(line[bi:ei]) + 1
                        mention = ""
                        bi = line.index('>') + 1
                        while line.find(CHARSEQ_END_TAG) < 0:
                            mention += line[bi:].strip() + ' '
                            bi = 0
                            line = f.readline()
                        ei = line.index(CHARSEQ_END_TAG)
                        mention += line[bi:ei].strip()
                        mention = mention.replace('  ', ' ').replace('&AMP;', '&').replace('&amp;', '&')
                        entity_annotation = EntityAnnotation(start, end, tag, mention)
                        entity_annotations.append(entity_annotation)

                if line.find(EXTENT_END_TAG) > -1:
                    extent_flag = False

            if line.find(ENTITY_END_TAG) > -1:
                extent_flag = False
                entity_flag = False

    entity_annotations.sort(key=lambda x: (x.start, x.end))

    index_map = {}
    with open(basename + '.sgm', 'r') as f:
        doc_org = f.read()

        doc_tmp = re.sub(r'<[^>]+>', '', doc_org)
        doc_tmp = re.sub(r'(\S+)\n(\S[^:])', r'\1 \2', doc_tmp)

        bi = doc_org.index(TEXT_BEGIN_TAG)
        ei = doc_org.index(TEXT_END_TAG) + len(TEXT_END_TAG)
        doc_modified = re.sub(r'<[^>]+>', '', doc_org[bi:ei])
        doc_modified = re.sub(r'(\S+)\n(\S[^:])', r'\1 \2', doc_modified)

        offset = doc_tmp.index(doc_modified)

        index = 0
        while len(doc_modified) > 0 and (doc_modified[0] == '\n' or doc_modified[0] == ' '):
            doc_modified = doc_modified[1:]
            offset += 1
        while index < len(doc_modified):
            while 0 < index < len(doc_modified) - 1 \
                    and (doc_modified[index-1] == '\n' or doc_modified[index-1] == ' ') \
                    and (doc_modified[index] == '\n' or doc_modified[index] == ' '):
                doc_modified = doc_modified[:index] + doc_modified[index+1:]
                offset += 1
            if doc_modified[index:index+5] == '&AMP;' or doc_modified[index:index+5] == '&amp;':
                doc_modified = doc_modified[:index] + '&' + doc_modified[index+5:]
                offset += 4
            index_map[index+offset] = index
            index += 1

    entity_annotations = [entity_annotation for entity_annotation in entity_annotations
                          if entity_annotation.start in index_map]

    for entity_annotation in entity_annotations:
        entity_annotation.start = index_map[entity_annotation.start]
        entity_annotation.end = index_map[entity_annotation.end]
        assert (entity_annotation.mention == doc_modified[entity_annotation.start:entity_annotation.end])

    sentences = tokenizer.tokenize(doc_modified)

    output_lines = []
    for sentence in sentences:

        tokens = list(sentence)

        words = list()
        for token in tokens:
            words.append(token.word)
        output_lines.append(' '.join(words) + '\n')

        s_start = tokens[0].begin
        s_end = tokens[-1].end
        labels = list()
        for entity_annotation in entity_annotations:
            if entity_annotation.start < s_start:
                continue
            elif s_end < entity_annotation.end:
                break
            label = Label()
            index = 0
            while index < len(tokens):
                token = tokens[index]
                if token.begin <= entity_annotation.start < token.end:
                    label.start = index
                    break
                index += 1
            while index < len(tokens):
                token = tokens[index]
                if token.begin < entity_annotation.end <= token.end:
                    label.end = index + 1
                    break
                index += 1

            assert (label.start is not None)
            assert (label.end is not None)
            label.tag = entity_annotation.type
            labels.append(label)
        output_lines.append('|'.join([label.__repr__() for label in labels]) + '\n')

        output_lines.append('\n')

        calc_stat(words, labels)

    return output_lines


def parse_ace_2004(tokenizer: Tokenizer) -> None:

    output_dir_path = "data/ace2004/"
    os.makedirs(output_dir_path, mode=0o755, exist_ok=True)

    output_file_list = ["ace2004.train", "ace2004.dev", "ace2004.test"]

    for split_info_file, output_file in zip(SPLIT_INFO_FILE_LIST, output_file_list):
        output_lines = []
        doc_count = 0
        sent_count = 0
        token_count = 0
        for tag in TAG_SET:
            TAG_SET[tag] = Stat()
        with open(SPLIT_INFO_DIR_PATH + split_info_file, 'r') as f:
            for line in f:
                basename = CORPUS_DIR_PATH + line.strip()
                output_lines_doc = parse_document(basename, tokenizer)
                output_lines.extend(output_lines_doc)
                doc_count += 1
                sent_count += len(output_lines_doc) // 3
                for depth in range(0, len(output_lines_doc), 3):
                    token_count += len(output_lines_doc[depth].split())

        with open(output_dir_path + output_file, 'w') as f:
            f.writelines(output_lines)

        print("")
        print("--- {}".format(output_file))
        print("# of documents:\t{:6d}".format(doc_count))
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
        print("total # of mentions:\t{}\t(layer:\t{},\tignored:\t{})".format(total, total_layer, total_ignored))
        for tag, stat in TAG_SET.items():
            print("\t{}:\t{:5d}\t(layer:\t{},\tignored:\t{})"
                  .format(tag, stat.total, stat.layer, stat.ignored))
        ave_labels = 0
        for _, stat in TAG_SET.items():
            ave_labels += stat.num_labels
        ave_labels /= token_count * len(TAG_SET)
        print("average # of labels:\t{:.2f}".format(ave_labels))


parse_ace_2004(Tokenizer())
