import os
import copy
import torch
import pickle
import numpy as np
import json
import random
import cv2
from PIL import Image
import sys
import tqdm
sys.path.append('./')
from libs.data.list_record_cache import ListRecordLoader
from libs.utils.vocab import TypeVocab, RelationVocab
from torchvision.transforms import functional as F
from libs.utils.tree_utils import read_synatxtree_file

class PickleLoader():
    
    def __init__(self, pickle_path, ly_vocab, re_vocab, mode, all_labels_path):
        assert mode in ['train', 'test']
        self.pickle_path = pickle_path
        self.mode = mode
        self.info = []
        self.syntax_tree = []
        self.info_path = pickle_path
        self.all_labels_path = all_labels_path
        self.init()
        self.ly_vocab = ly_vocab
        self.re_vocab = re_vocab
        


    def init(self):
        with open(self.info_path) as f:
            self.info = json.load(f)

        if self.mode == 'test':
            self.test_synatx_tree = [None for _ in range(len(self.info))]
            all_synatx_trees, all_pdf_paths = read_synatxtree_file(self.all_labels_path)
            legal_products = [p['product'] for p in self.info]
            for pdf_path, synatx_tree in zip(all_pdf_paths, all_synatx_trees):
                product = os.path.basename(pdf_path)
                if product in legal_products:
                    idx = legal_products.index(product)
                    self.test_synatx_tree[idx] = synatx_tree
            assert None not in self.test_synatx_tree

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        data = self.info[idx]
        img_lst = []
        for img_path in data['imgs_path']:
            img = cv2.imread(img_path)
            img_lst.append(img)
        data['imgs'] = img_lst
        encoder_input, ly, re, align, transcripts, bboxes, stride = self.cal_items(data)
        if re == []:
            print('re==[] when idx =', idx, data['pdf_path'])
            return self[random.randint(0, len(self) - 1)]
        if self.mode == 'train':
            return dict(
                idx=idx,
                ly=ly,
                re=re,
                align=align,
                bboxes=bboxes,
                transcripts=transcripts,
                encoder_input=encoder_input,
                stride=stride,
                pdf_path=data['pdf_path']
            )
        else:
            return dict(
                idx=idx,
                ly=ly,
                re=re,
                align=align,
                bboxes=bboxes,
                transcripts=transcripts,
                encoder_input=encoder_input,
                stride=stride,
                lines=data['lines'],
                pdf_path=data['pdf_path'],
                synatx_tree=self.test_synatx_tree[idx]
            )            

    def cal_items(self, word):

        stride = word['stride']

        ly, re, align, transcripts, bboxes = [], [], [], [], []
        re_ids_map = dict()
        re_ids_map[0] = 0
        index = 1
        for lines_pg in word['lines']:
            bboxes_pg = []
            transcripts_pg = []
            for line in lines_pg:

                bboxes_pg.append([stride * item for item in line['box']]) # the stride between feature map and images
                transcripts_pg.append(line['content'])
                ly.append(self.ly_vocab.title_id if line['is_title'] else self.ly_vocab.line_id)

                if line['is_title']:
                    re.append(self.re_vocab._words_ids_map[line['relation']])
                    align.append(re_ids_map[line['parent']])
                    re_ids_map[line['line_id']] = index
                    index += 1

            bboxes.append(bboxes_pg)
            transcripts.append(transcripts_pg)
            
            assert sum([item==self.ly_vocab.title_id for item in ly]) == len(re) == len(align) == len(re_ids_map) - 1

        imgs, bboxes = self.random_scale(word['imgs'], bboxes)

        encoder_input = list()
        for image in imgs:
            image = F.to_tensor(image)
            image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)
            encoder_input.append(image)

        return encoder_input, ly, re, align, transcripts, bboxes, stride

    def random_scale(self, imgs, boxes)   :
        if self.mode == 'test':
            return imgs, boxes
        scale = random.choice([0.8, 0.9, 1, 1.1, 1.2])
        imgs = [cv2.resize(image, None, fx=scale, fy=scale) for image in imgs]
        boxes = [[[cor * scale for cor in box]for box in box_pg]for box_pg in boxes]
        return imgs, boxes
        
        

class InvalidFormat(Exception):
    pass


class LRCRecordLoader:
    def __init__(self, lrc_path):
        self.loader = ListRecordLoader(lrc_path)

    def __len__(self):
        return len(self.loader)

    def get_data(self, idx):
        word = self.loader.get_record(idx)
        return word


class Dataset:
    def __init__(self, loader, ly_vocab, re_vocab):
        self.loader = loader
        self.ly_vocab = ly_vocab
        self.re_vocab = re_vocab


    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def get_info(self, idx):
        loader, rela_idx = self._match_loader(idx)
        return loader.get_data(rela_idx)
        
    def cal_items(self, word):
        encoder_input = list()
        for image in word['imgs']: # [cv2.imwrite('%d.png' % i, word['imgs'][i]) for i in range(len(word['imgs']))]
            image = F.to_tensor(image)
            image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)
            encoder_input.append(image)

        stride = word['stride']

        ly, re, align, transcripts, bboxes = [], [], [], [], []
        re_ids_map = dict()
        re_ids_map[0] = 0
        index = 1
        for lines_pg in word['lines']:
            bboxes_pg = []
            transcripts_pg = []
            for line in lines_pg:

                bboxes_pg.append([stride * item for item in line['bbox']]) # the stride between feature map and images
                transcripts_pg.append(line['content'])
                ly.append(self.ly_vocab.title_id if line['is_title'] else self.ly_vocab.line_id)

                if line['is_title']:
                    re.append(self.re_vocab._words_ids_map[line['relation']])
                    align.append(re_ids_map[line['parent']])
                    re_ids_map[line['line_id']] = index
                    index += 1

            bboxes.append(bboxes_pg)
            transcripts.append(transcripts_pg)
            
            assert sum([item==self.ly_vocab.title_id for item in ly]) == len(re) == len(align) == len(re_ids_map) - 1

        return encoder_input, ly, re, align, transcripts, bboxes, stride

    def __getitem__(self, idx):
        try:
            # idx = 1
            loader, rela_idx = self._match_loader(idx)
            word = loader.get_data(rela_idx)
            encoder_input, ly, re, align, transcripts, bboxes, stride = self.cal_items(word)
            return dict(
                idx=idx,
                ly=ly,
                re=re,
                align=align,
                bboxes=bboxes,
                transcripts=transcripts,
                encoder_input=encoder_input,
                stride=stride
            )
        except Exception as e:
            print('Error occured while load data: %d' % idx)
            raise e
        

def collate_func(batch_data):
    batch_size = len(batch_data)
    input_channels = batch_data[0]['encoder_input'][0].shape[0]

    max_H = max([max([page.shape[1] for page in data['encoder_input']]) for data in batch_data])
    max_W = max([max([page.shape[2] for page in data['encoder_input']]) for data in batch_data])
    max_ly_len = max([len(data['ly']) for data in batch_data])
    max_re_len = max([len(data['re']) for data in batch_data])
    max_align_len = max([len(data['align']) for data in batch_data])

    batch_encoder_input = []
    batch_encoder_input_mask = []
    batch_iamge_size = []

    batch_ly = torch.zeros(batch_size, max_ly_len).to(torch.long)
    batch_re = torch.zeros(batch_size, max_re_len).to(torch.long)
    batch_ma = torch.zeros(batch_size, max_align_len).to(torch.long)

    batch_ly_mask = torch.zeros(batch_size, max_ly_len).to(torch.float32)
    batch_re_mask = torch.zeros(batch_size, max_re_len).to(torch.float32)
    batch_ma_mask = torch.zeros(batch_size, max_align_len).to(torch.float32)

    batch_transcripts = []
    batch_bboxes = []
    for batch_idx, data in enumerate(batch_data):

        stride = data['stride']
        encoder_input = torch.zeros(len(data['encoder_input']), input_channels, max_H, max_W).to(torch.float32)
        encoder_input_mask = torch.zeros(len(data['encoder_input']), 1, max_H, max_W).to(torch.float32)
        iamge_size = []
        
        for page_id, encoder_input_page in enumerate(data['encoder_input']):
            encoder_input[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = encoder_input_page
            encoder_input_mask[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = 1.
            image_H = int(encoder_input_page.shape[1] / stride)
            image_W = int(encoder_input_page.shape[2] / stride)
            iamge_size.append((image_H, image_W))

        batch_encoder_input.append(encoder_input)
        batch_encoder_input_mask.append(encoder_input_mask)
        batch_iamge_size.append(iamge_size)

        ly_len = len(data['ly'])
        re_len = len(data['re'])
        ma_len = len(data['align'])
        batch_ly[batch_idx, :ly_len] = torch.tensor(data['ly']).to(torch.long)
        batch_re[batch_idx, :re_len] = torch.tensor(data['re']).to(torch.long)
        batch_ma[batch_idx, :ma_len] = torch.tensor(data['align']).to(torch.long)
        batch_ly_mask[batch_idx, :ly_len] = 1.
        batch_re_mask[batch_idx, :re_len] = 1.
        batch_ma_mask[batch_idx, :ma_len] = 1.

        batch_transcripts.append(data['transcripts'])
        batch_bboxes.append(data['bboxes'])

    return dict(
        encoder_input=batch_encoder_input,
        encoder_input_mask=batch_encoder_input_mask,
        bboxes=batch_bboxes,
        transcripts=batch_transcripts,
        image_size=batch_iamge_size,
        ly=batch_ly,
        ly_mask=batch_ly_mask,
        re=batch_re,
        re_mask=batch_re_mask,
        ma=batch_ma,
        ma_mask=batch_ma_mask,
    )


class ValidDataset:
    def __init__(self, loaders, ly_vocab):
        self.loaders = loaders
        self.ly_vocab = ly_vocab

    def _match_loader(self, idx):
        offset = 0
        for loader in self.loaders:
            if len(loader) + offset > idx:
                return loader, idx - offset
            else:
                offset += len(loader)
        raise IndexError()

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def cal_items(self, word):
        encoder_input = list()
        for image in word['imgs']:
            image = F.to_tensor(image)
            image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)
            encoder_input.append(image)

        stride = word['stride']

        transcripts, bboxes = [], []
        for lines_pg in word['lines']:
            bboxes_pg = []
            transcripts_pg = []
            for line in lines_pg:
                bboxes_pg.append([int(stride * item) for item in line['bbox']])
                transcripts_pg.append(line['content'])
            bboxes.append(bboxes_pg)
            transcripts.append(transcripts_pg)

        return encoder_input, transcripts, bboxes, stride

    def __getitem__(self, idx):
        try:
            loader, rela_idx = self._match_loader(idx)
            word = loader.get_data(rela_idx)
            encoder_input, transcripts, bboxes, stride = self.cal_items(word)
            return dict(
                idx=idx,
                bboxes=bboxes,
                transcripts=transcripts,
                encoder_input=encoder_input,
                stride=stride,
                word=word
            )
        except Exception as e:
            print('Error occured while load data: %d' % idx)
            raise e
        

def valid_collate_func(batch_data):
    batch_size = len(batch_data)
    input_channels = batch_data[0]['encoder_input'][0].shape[0]

    max_H = max([max([page.shape[1] for page in data['encoder_input']]) for data in batch_data])
    max_W = max([max([page.shape[2] for page in data['encoder_input']]) for data in batch_data])
    max_re_len = max([len(data['re']) for data in batch_data])
    max_align_len = max([len(data['align']) for data in batch_data])    

    batch_encoder_input = []
    batch_encoder_input_mask = []
    batch_iamge_size = []

    batch_transcripts = []
    batch_bboxes = []
    batch_lines = []

    pdf_paths = []
    synatx_trees = []
    lys = []
    batch_re = torch.zeros(batch_size, max_re_len).to(torch.long)
    batch_ma = torch.zeros(batch_size, max_align_len).to(torch.long)

    for batch_idx, data in enumerate(batch_data):

        stride = data['stride']
        encoder_input = torch.zeros(len(data['encoder_input']), input_channels, max_H, max_W).to(torch.float32)
        encoder_input_mask = torch.zeros(len(data['encoder_input']), 1, max_H, max_W).to(torch.float32)
        iamge_size = []
        
        for page_id, encoder_input_page in enumerate(data['encoder_input']):
            encoder_input[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = encoder_input_page
            encoder_input_mask[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = 1.
            image_H = int(encoder_input_page.shape[1] / stride)
            image_W = int(encoder_input_page.shape[2] / stride)
            iamge_size.append((image_H, image_W))

        batch_encoder_input.append(encoder_input)
        batch_encoder_input_mask.append(encoder_input_mask)
        batch_iamge_size.append(iamge_size)

        batch_transcripts.append(data['transcripts'])
        batch_bboxes.append(data['bboxes'])
        batch_lines.append(data['lines'])
        pdf_paths.append(data['pdf_path'])
        synatx_trees.append(data['synatx_tree'])
        lys.append(data['ly'])

        re_len = len(data['re'])
        ma_len = len(data['align'])
        batch_re[batch_idx, :re_len] = torch.tensor(data['re']).to(torch.long)
        batch_ma[batch_idx, :ma_len] = torch.tensor(data['align']).to(torch.long)

    return dict(
        encoder_input=batch_encoder_input,
        encoder_input_mask=batch_encoder_input_mask,
        bboxes=batch_bboxes,
        transcripts=batch_transcripts,
        image_size=batch_iamge_size,
        lines=batch_lines,
        pdf_paths=pdf_paths,
        synatx_trees=synatx_trees,
        lys=lys,
        res=batch_re,
        mas=batch_ma,
    )