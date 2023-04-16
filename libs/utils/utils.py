from json import encoder
import os
import json
import torch
import numpy as np
from .metric import AccMetric, AccMulMetric


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='gbk')
        else:
            return super(NpEncoder, self).default(obj)
            

def cal_mean_lr(optimizer):
    lrs = [group['lr'] for group in optimizer.param_groups]
    return sum(lrs)/len(lrs)


def cal_total_acc(pred_result, ly_cls_labels, ly_labels_mask, ly_vocab):
    ly_cls_preds, (re_cls_preds, ma_att_preds) = pred_result

    acc_metric = AccMetric()
    
    # ly acc
    cls_correct, cls_total = acc_metric(ly_cls_preds, ly_cls_labels, ly_labels_mask == 1)
    cls_line_correct, cls_line_total = acc_metric(ly_cls_preds, ly_cls_labels, (ly_labels_mask == 1) & (ly_cls_labels==ly_vocab.line_id))
    cls_title_correct, cls_title_total = acc_metric(ly_cls_preds, ly_cls_labels, (ly_labels_mask == 1) & (ly_cls_labels==ly_vocab.title_id))

    # acc info
    acc_info = dict()
    acc_info['cls_acc'] = cls_correct / cls_total
    acc_info['cls_line_acc'] = cls_line_correct / cls_line_total
    acc_info['cls_title_acc'] = cls_title_correct / cls_title_total

    return acc_info



def integrate_predicts(word, ly_cls_preds, re_cls_preds, ma_att_preds, ly_vocab, re_vocab):
    lines = [line for lines_pp in word['lines'] for line in lines_pp]
    infos = list()
    title_index = 0
    for line_index, line in enumerate(lines):
        info = dict(bbox=line['bbox'], content=line['content'], char_size=line['char_size'], \
            char_type=line['char_type'], page_id=line['page_id'])
        if ly_cls_preds[line_index] == ly_vocab.title_id:
            info['is_title'] = True
            info['relation'] = re_vocab.id_to_word(int(re_cls_preds[title_index]))
            info['parent_id'] = int(torch.argmax(ma_att_preds[title_index, :title_index+1]))
            title_index += 1
        else:
            info['is_title'] = False
            info['relation'] = None
            info['parent_id'] = None
        infos.append(info)
    infos = split_list(infos, [len(lines_pp) for lines_pp in word['lines']])
    return infos


def split_list(init_list, split_len):
    splited_list = list()
    start_index = 0
    for len in split_len:
        splited_list.append(init_list[start_index:start_index+len])
        start_index += len
    return splited_list

        