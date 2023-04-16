import sys
sys.path.append('./')
import os
import re
import math
import torch
import numpy as np
import glob
import re
from libs.utils.syntax_tree import SyntaxTree


def get_filepaths(file_dir, ext='.pdf'):
    if type(ext) == str:
        ext = [ext]
    all_files = []
    for sigle_ext in ext:
        single_extfiles = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[-1].lower() == sigle_ext:
                    single_extfiles.append(root+"/"+file)
        all_files.extend(single_extfiles)
    
    return all_files



def decoder_restore(pdf_path, lines, ly_cls_preds_pb, ly_vocab, ly_re, re_cls_preds_pb, ma_att_preds_pb, stride):
    title_idxs = torch.where(ly_cls_preds_pb == ly_vocab.title_id)[0]
    flatten_lines = [l for page in lines for l in page]
    title_lines = [flatten_lines[idx] for idx in title_idxs]

    title_info_lst = [{'tid':'', 'children_idx':[], 'key':''} for _ in range(len(title_lines))]
    parent_idx_lst = [torch.argmax(ma_att_preds_pb[t][:t + 1]).item() - 1 for t in range(len(ma_att_preds_pb))]
    re_cls_preds_pb = re_cls_preds_pb.tolist()
    for time_t, (relation, parent_idx) in enumerate(zip(re_cls_preds_pb, parent_idx_lst)):
        cur_line = title_lines[time_t]
        if parent_idx == -1:
            title_info_lst[time_t]['tid'] = '1'
            title_info_lst[time_t]['key'] = cur_line['content']
        else:
            parent_info = title_info_lst[parent_idx]
            parent_tid = parent_info['tid']
            if ly_re.id_to_word(relation) == 'equal':
                tid = '.'.join(parent_tid.split('.')[:-1] + [str(eval(parent_tid.split('.')[-1]) + 1)])
                title_info_lst[time_t]['tid'] = tid
                title_info_lst[time_t]['key'] = cur_line['content']
            elif ly_re.id_to_word(relation) == 'contain':
                tid = parent_tid + '.' + str(len(parent_info['children_idx']) + 1)
                title_info_lst[time_t]['tid'] = tid
                title_info_lst[time_t]['key'] = cur_line['content']
                parent_info['children_idx'].append(time_t)
            elif ly_re.id_to_word(relation) == 'sibling':
                tid = parent_tid
                title_info_lst[time_t]['tid'] = tid
                title_info_lst[time_t]['key'] = cur_line['content']
    # sibling
    to_delete_idx_lst = []
    for cur_i, title_info in enumerate(title_info_lst):
        if cur_i not in to_delete_idx_lst:
            for search_j, search_info in enumerate(title_info_lst):
                if search_j > cur_i and search_j not in to_delete_idx_lst:
                    if title_info['tid'] == search_info['tid']:
                        title_info['key'] += search_info['key']
                        to_delete_idx_lst.append(search_j)
    title_info_lst = [t for t_i, t in enumerate(title_info_lst) if t_i not in to_delete_idx_lst]
    pred_tree = SyntaxTree.read_hier(title_info_lst, pdf_path)
    return pred_tree 

data_post_process_vis = decoder_restore
