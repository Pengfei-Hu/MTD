import Levenshtein


from apted import APTED, Config
from apted.helpers import Tree

import sys
sys.path.append('./')

from libs.utils.utils_dataset import repr_latex, repr_pdf
from libs.utils.tree_utils import read_synatxtree_file_debug

__all__ = ['tree_edit_distance']


def cal_wer(pred_str, label_str):
    if len(label_str):
        return Levenshtein.distance(pred_str, label_str) / len(label_str)
    else:
        return 1e6

class HuConfig(Config):

    def rename(self, node1, node2):
        """node1:pred, node2:label"""
        str_1, str_2 = repr_pdf(node1.name).replace(' ', '').lower(), repr_latex(node2.name).replace(' ', '').lower()
        if cal_wer(str_1, str_2) < 0.2:
            return 0
        else:
            return 1

def tree_edit_distance(pred_tree, true_tree):
    distance = APTED(pred_tree.root, true_tree.root, HuConfig()).compute_edit_distance()
    teds = 1.0 - (float(distance) / max([len(pred_tree), len(true_tree)]))
    return teds

if __name__ == "__main__":
    (pred_tree, label_tree), pdf_paths = read_synatxtree_file_debug('libs/utils/debug_tree.log')
    socre = tree_edit_distance(pred_tree, label_tree)
    dev = 233

