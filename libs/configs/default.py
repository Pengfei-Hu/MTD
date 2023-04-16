import os
import torch
from libs.utils.vocab import TypeVocab, RelationVocab
from transformers import BertTokenizer, BertModel
from .extractor.model import Model as Extractor

device = torch.device('cuda')

# dataloader 
train_batch_size = 1
train_num_workers = 0
train_pickle_path = "dataset/train.json"

valid_batch_size = 1
valid_num_workers = 0
valid_pickle_path = "dataset/test.json"

all_labels_path = 'dataset/all_labels.log'

# vocab
ly_vocab = TypeVocab()
re_vocab = RelationVocab()

# extractor
extractor = Extractor().eval()

# bert
tokenizer = BertTokenizer.from_pretrained("Your_path_to/English_base_cased/")
bert = BertModel.from_pretrained("Your_Path_to/English_base_cased/").eval()

# encoder
in_dim = 64
encoder_layers = [1, 1]
encoder_dim = 128
scale = 1.0
pool_size = (3,3)
word_dim = 768

# decoder
embed_dim = 128
feat_dim = 128
lm_state_dim = 128
proj_dim = 128
cover_kernel = 7

# train params
base_lr = 0.0005
min_lr = 1e-6
weight_decay = 0

num_epochs = 30
sync_rate = 20
valid_epoch = 1

log_sep = 5
cache_nums = 1000

work_dir = './experiments/default'

train_checkpoint = None
eval_checkpoint = os.path.join(work_dir, 'best_TEDS_model.pth')