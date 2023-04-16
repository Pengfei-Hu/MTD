import torch
from torch import nn
from .loss import sigmoid_focal_loss
from libs.utils.metric import AccMetric


class Classifier(nn.Module):

    def __init__(self, ly_vocab, feat_dim):
        super().__init__()
        self.ly_vocab = ly_vocab
        self.feat_dim = feat_dim

        self.ly_cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim//2),
            nn.Tanh(),
            nn.Linear(self.feat_dim//2, len(self.ly_vocab))
        )
    
    def forward(self, feats, feats_mask, ly_cls_labels=None, ly_labels_mask=None):

        loss_cache = dict()
        ly_cls_logits = self.ly_cls(feats)
        ly_cls_preds = torch.argmax(ly_cls_logits, dim=-1).detach()
        
        if self.training:
            valid_length = torch.sum(ly_labels_mask == 1, dim=1).detach()
            ly_onehot_labels = torch.zeros(ly_cls_labels.shape[0], ly_cls_labels.shape[1], len(self.ly_vocab)).to(ly_cls_labels).scatter(2, ly_cls_labels.unsqueeze(-1), 1)

            ly_cls_loss = sigmoid_focal_loss(ly_cls_logits, ly_onehot_labels, reduction='none').sum(-1) * ly_labels_mask
            ly_cls_loss = torch.mean(ly_cls_loss.sum(-1) / valid_length)
            loss_cache['ly_cls_loss'] = ly_cls_loss

            acc_metric = AccMetric()
            
            # ly acc
            cls_correct, cls_total = acc_metric(ly_cls_preds, ly_cls_labels, ly_labels_mask == 1)
            cls_line_correct, cls_line_total = acc_metric(ly_cls_preds, ly_cls_labels, (ly_labels_mask == 1) & (ly_cls_labels==self.ly_vocab.line_id))
            cls_title_correct, cls_title_total = acc_metric(ly_cls_preds, ly_cls_labels, (ly_labels_mask == 1) & (ly_cls_labels==self.ly_vocab.title_id))

            loss_cache['cls_acc'] = cls_correct / cls_total
            loss_cache['cls_line_acc'] = cls_line_correct / cls_line_total
            loss_cache['cls_title_acc'] = cls_title_correct / cls_title_total

        return ly_cls_preds, loss_cache


def build_classifier(cfg):
    classifier = Classifier(
        ly_vocab=cfg.ly_vocab,
        feat_dim=cfg.feat_dim
    )
    return classifier