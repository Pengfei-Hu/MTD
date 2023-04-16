from torch import nn
from .utils import align_feats
from libs.model.encoder import build_encoder
from libs.model.classifier import build_classifier
from libs.model.decoder import build_decoder
from .encoder import LstmFusion


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        self.classifier = build_classifier(cfg)
        self.title_encoder = LstmFusion(cfg.embed_dim, cfg.embed_dim // 2)
        self.decoder = build_decoder(cfg)
        
    def forward(self, encoder_input, encoder_input_mask, image_size, transcripts, encoder_input_bboxes, extractor, tokenizer, bert, ly=None, ly_mask=None, re=None, re_mask=None, ma=None, ma_mask=None):

        feats, feats_mask = self.encoder(encoder_input, encoder_input_mask, image_size, transcripts, encoder_input_bboxes, extractor, tokenizer, bert)

        result_info = dict()

        # cls  title
        ly_cls_preds, ly_cls_info = self.classifier(feats, feats_mask, ly, ly_mask)
        result_info.update(ly_cls_info)

        # extract title feats
        if ly is not None:
            title_mask = ly == self.cfg.ly_vocab.title_id
            title_feats = [feats[batch_idx, title_mask_pb, :] for batch_idx, title_mask_pb in enumerate(title_mask)]
        else:
            title_mask = ly_cls_preds == self.cfg.ly_vocab.title_id
            title_feats = [feats[batch_idx, title_mask_pb, :] for batch_idx, title_mask_pb in enumerate(title_mask)]

        title_feats, title_feats_mask = align_feats(title_feats) # (B, N, C)

        # encode head feats
        title_feats = self.title_encoder(title_feats, title_feats_mask)

        title_feats = title_feats.permute(0,2,1).contiguous() # (B, C, N)

        # decode structure of title
        if self.training:
            de_info = self.decoder(title_feats, title_feats_mask, re, re_mask, ma, ma_mask)
            result_info.update(de_info)
            return result_info
        else:
            de_results = self.decoder(title_feats, title_feats_mask)
            return ly_cls_preds, de_results

