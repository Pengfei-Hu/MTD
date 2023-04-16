import os
from .syntax_tree import SyntaxTree

def read_synatxtree_file(synatxtree_log_path):
    synatxtree_lst, pdf_paths = [], []
    with open(synatxtree_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        boundary_idx_lst = []
        for l_i, l in enumerate(lines):
            if len(l) - len(l.replace('***', '')) == 6:
                boundary_idx_lst.append(l_i)
        for idx_i, idx in enumerate(boundary_idx_lst):
            l = lines[idx]
            sub_paths = l.replace('***', '').split('\\')
            for s_i, sp in enumerate(sub_paths):
                if sp.endswith('_latex'):
                    break
            pdf_path = '/'.join(sub_paths[:s_i + 1])[:-6] + '.pdf'
            pdf_paths.append(pdf_path)
            if idx_i == len(boundary_idx_lst) - 1:
                end = len(lines)
            else:
                end = boundary_idx_lst[idx_i + 1]
            tree_info = lines[idx:end]
            tree = SyntaxTree.read_log(tree_info)
            synatxtree_lst.append(tree)
    return synatxtree_lst, pdf_paths

def read_synatxtree_file_debug(synatxtree_log_path):
    synatxtree_lst, pdf_paths = [], []
    with open(synatxtree_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l[2:-3] for l in lines]
        boundary_idx_lst = []
        for l_i, l in enumerate(lines):
            if len(l) - len(l.replace('***', '')) == 6:
                boundary_idx_lst.append(l_i)
        for idx_i, idx in enumerate(boundary_idx_lst):
            l = lines[idx]
            sub_paths = l.replace('***', '').split('\\')
            for s_i, sp in enumerate(sub_paths):
                if sp.endswith('_latex'):
                    break
            pdf_path = '/'.join(sub_paths[:s_i + 1])[:-6] + '.pdf'
            pdf_paths.append(pdf_path)
            if idx_i == len(boundary_idx_lst) - 1:
                end = len(lines)
            else:
                end = boundary_idx_lst[idx_i + 1]
            tree_info = lines[idx:end]
            tree = SyntaxTree.read_log(tree_info)
            synatxtree_lst.append(tree)
    return synatxtree_lst, pdf_paths    