import torch


class AccMetric:
    def __call__(self, preds, labels, labels_mask):
        mask = (labels_mask != 0) & (labels != -1)
        correct_nums = float(torch.sum((preds == labels) & mask).detach().cpu().item())
        total_nums = max(float(torch.sum(mask).detach().cpu().item()), 1e-6)
        return correct_nums, total_nums

def ly_prf(preds, labels):
    label_title_idxs = labels == 0
    label_num = label_title_idxs.sum().detach().cpu().item()
    pred_title_idxs = preds == 0
    pred_num = pred_title_idxs.sum().detach().cpu().item()
    p_up = torch.logical_and(label_title_idxs, pred_title_idxs).sum().detach().cpu().item()
    if pred_num == 0:
        p = 0
    else:
        p = p_up / pred_num
    if label_num == 0:
        r = 0
    else:
        r = p_up / label_num
    if p == 0 or r == 0:
        f = 0
    else:
        f = 2 / (1 / p + 1/ r)
    return p, r, f



class AccMulMetric:
    def __call__(self, preds, labels, labels_mask):
        mask = labels_mask != 0
        correct_nums = float(torch.sum((preds == labels).min(1)[0] & mask).detach().cpu().item())
        total_nums = max(float(torch.sum(mask).detach().cpu().item()), 1e-6)
        return correct_nums, total_nums