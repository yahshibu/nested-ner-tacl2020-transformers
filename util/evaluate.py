from typing import List, Tuple
from collections import defaultdict


def evaluate(gold_entities: List[List[Tuple[int, int, int]]], pred_entities: List[List[Tuple[int, int, int]]]) \
        -> Tuple[int, int, int, int]:
    prec_all_num, prec_num, recall_all_num, recall_num = 0, 0, 0, 0
    for g_ets, p_ets in zip(gold_entities, pred_entities):
        recall_all_num += len(g_ets)
        prec_all_num += len(p_ets)

        for et in g_ets:
            if et in p_ets:
                recall_num += 1

        for et in p_ets:
            if et in g_ets:
                prec_num += 1

    return prec_all_num, prec_num, recall_all_num, recall_num


def evaluate_detail(gold_entities: List[List[Tuple[int, int, int]]], pred_entities: List[List[Tuple[int, int, int]]]) \
        -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    prec_all_num_n, prec_num_n, recall_all_num_n, recall_num_n = 0, 0, 0, 0
    prec_all_num_o, prec_num_o, recall_all_num_o, recall_num_o = 0, 0, 0, 0
    for g_ets, p_ets in zip(gold_entities, pred_entities):
        if if_overlap(g_ets):
            recall_all_num_o += len(g_ets)
            prec_all_num_o += len(p_ets)

            for et in g_ets:
                if et in p_ets:
                    recall_num_o += 1

            for et in p_ets:
                if et in g_ets:
                    prec_num_o += 1
        else:
            recall_all_num_n += len(g_ets)
            prec_all_num_n += len(p_ets)

            for et in g_ets:
                if et in p_ets:
                    recall_num_n += 1

            for et in p_ets:
                if et in g_ets:
                    prec_num_n += 1

    return (prec_all_num_n, prec_num_n, recall_all_num_n, recall_num_n), \
           (prec_all_num_o, prec_num_o, recall_all_num_o, recall_num_o)


def if_overlap(ets: List[Tuple[int, int, int]]) -> bool:
    len_ets = len(ets)
    if len_ets == 0:
        return False
    for i in range(len_ets-1):
        candidates = ets[i+1:]
        focus = ets[i]
        for cand in candidates:
            if (focus[0] < cand[0] < focus[1] < cand[1]) or \
                    (focus[0] <= cand[0] and cand[1] <= focus[1]) or \
                    (cand[0] <= focus[0] and focus[1] <= cand[1]):
                return True
        return False


def count_overlap(entities: List[List[Tuple[int, int, int]]]) -> Tuple[int, int]:
    """
    Counting the # of crossing structures
    Naive way
    """
    num_cross = 0
    num_nest = 0
    for ets in entities:
        len_ets = len(ets)
        if len_ets == 0:
            continue
        for i in range(len_ets-1):
            candidates = ets[i+1:]
            focus = ets[i]
            for cand in candidates:
                if focus[0] < cand[0] < focus[1] < cand[1]:
                    num_cross += 1
                if (focus[0] <= cand[0] and cand[1] <= focus[1]) or (cand[0] <= focus[0] and focus[1] <= cand[1]):
                    num_nest += 1
    return num_cross, num_nest


def detail_count_overlap(g_entities: List[List[Tuple[int, int, int]]], p_entities: List[List[Tuple[int, int, int]]]) \
        -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Counting the # of crossing structures
    Naive way
    """
    num_all = 0
    num_left = 0
    num_right = 0
    num_other = 0
    c_num_left = 0
    c_num_right = 0
    c_num_other = 0

    for ets, p_ets in zip(g_entities, p_entities):
        len_ets = len(ets)
        if len_ets == 0:
            continue
        for i in range(len_ets-1):
            num_all += 1
            candidates = ets[i+1:]
            focus = ets[i]
            for cand in candidates:
                if cand[0] == focus[0] and cand[1] != focus[1] and cand[2] == focus[2]:
                    num_left += 1
                    if cand in p_ets and focus in p_ets:
                        c_num_left += 1
                elif cand[1] == focus[1] and cand[0] != focus[0] and cand[2] == focus[2]:
                    num_right += 1
                    if cand in p_ets and focus in p_ets:
                        c_num_right += 1
                else:
                    num_other += 1
                    if cand in p_ets and focus in p_ets:
                        c_num_other += 1
    return (num_left, c_num_left), (num_right, c_num_right), (num_other, c_num_other)


def detail_count_overlap_b(g_entities: List[List[Tuple[int, int, int]]], p_entities: List[List[Tuple[int, int, int]]]) \
        -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Counting the # of crossing structures
    # of the mentions with the same boudaries
    """
    num_left = 0
    num_right = 0
    num_other = 0
    c_num_left = 0
    c_num_right = 0
    c_num_other = 0

    for ets, p_ets in zip(g_entities, p_entities):
        len_ets = len(ets)
        if len_ets == 0:
            continue

        start_dic = defaultdict(list)
        end_dic = defaultdict(list)
        for e in ets:
            start_dic[(e[0], e[2])].append(e)
            end_dic[(e[1], e[2])].append(e)

        for k, v in start_dic.items():
            if len(v) > 1:
                num_left += len(v)
                for e in v:
                    if e in p_ets:
                        c_num_left += 1

        for k, v in end_dic.items():
            if len(v) > 1:
                num_right += len(v)
                for e in v:
                    if e in p_ets:
                        c_num_right += 1

    return (num_left, c_num_left), (num_right, c_num_right), (num_other, c_num_other)


if __name__ == "__main__":
    num = count_overlap([[(0, 2, 2), (1, 3, 1)]])
    print(num)
