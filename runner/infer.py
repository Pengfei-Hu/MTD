import sys
sys.path.append('./')
sys.path.append('../')
import os
import json
import tqdm
import torch
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.checkpoint import load_checkpoint
from libs.utils.comm import synchronize
from libs.utils.post_process import data_post_process_vis as post_process_visual
from libs.utils.teds import tree_edit_distance
from libs.utils.metric import ly_prf


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='default')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--visual_pretrain_weights", type=str, default='./visual_pretrain_weights.pth')
    args = parser.parse_args()
    
    setup_config(args.cfg)

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('MTD', cfg.work_dir, 'train.log')
    logger.info('Use config: %s' % args.cfg)
    if os.path.exists(args.visual_pretrain_weights):
        cfg.extractor.init_weights(args.visual_pretrain_weights)
        logger.info('Visual module load weights from %s' % args.visual_pretrain_weights)
    else:
        logger.info('Visual module does not load weights')


def cal_f_measure(ly_pred, ly_label, re_pred, ma_pred_pb, re_label, ma_label):

    ly_pred_idx = torch.where((ly_pred == 0))[0]
    ly_label_idx = torch.where((ly_label == 0))[0]
    pred_tuple_lst = [] 
    label_tuple_lst = []
    ma_pred_idx = [torch.argmax(ma_pred_pb[t][:t + 1]).item() - 1 for t in range(len(ma_pred_pb))]
    ma_lable_idx = (ma_label - 1).tolist()

    for title_i, (ref_idx, relation) in enumerate(zip(ma_pred_idx, re_pred)):
        idx_token = ly_pred_idx[title_i]
        if ref_idx == -1:
            idx_ref = -1 # ROOT
        else:
            idx_ref = ly_pred_idx[ref_idx]
        pred_tuple_lst.append([int(idx_token), int(idx_ref), int(relation)])

    for title_i, (ref_idx, relation) in enumerate(zip(ma_lable_idx, re_label)):
        idx_token = ly_label_idx[title_i]
        if ref_idx == -1:
            idx_ref = -1 # ROOT
        else:
            idx_ref = ly_label_idx[ref_idx]
        label_tuple_lst.append([int(idx_token), int(idx_ref), int(relation)])  

    equal = lambda x, y:x[0] == y[0] and x[1] == y[1] and x[2] == y[2]
    p_num = len(pred_tuple_lst)
    correct_num = 0
    for pred in pred_tuple_lst:
        for label in label_tuple_lst:
            if equal(label, pred):
                correct_num += 1
                break
    if p_num > 0:
        p = correct_num / p_num
    else:
        p = 0
    recall_num = 0
    for label in label_tuple_lst:
        for pred in pred_tuple_lst:
            if equal(label, pred):
                recall_num += 1
                break
        debug = 233
    l_num = len(label_tuple_lst)    
    if l_num > 0:
        r = recall_num / l_num
    else:
        r = 0

    if p == 0 or r == 0:
        f = 0
    else:
        f = 2 / (1 / p + 1/ r)
    return p, r, f

def valid(cfg, dataloader, model, epoch):
    model.eval()
    tokenizer = cfg.tokenizer
    extractor = cfg.extractor.to(cfg.device)
    bert = cfg.bert.to(cfg.device)
    all_predicted_result = list()
    p_lst, r_lst, f_lst = [], [], []
    p_ma_lst, r_ma_lst, f_ma_lst = [], [], []
    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        try:
            encoder_input = [data.to(cfg.device) for data in data_batch['encoder_input']]
            encoder_input_mask = [data.to(cfg.device) for data in data_batch['encoder_input_mask']]
            encoder_input_bboxes = [[torch.tensor(page).to(cfg.device).float() for page in data] for data in data_batch['bboxes']]
            transcripts = data_batch['transcripts']
            image_size = data_batch['image_size']
            pdf_paths = data_batch['pdf_paths']
            batch_lines = data_batch['lines']
            batch_label_synatx_tree = data_batch['synatx_trees']
            batch_label_lys = torch.tensor(data_batch['lys']).to(cfg.device)
            label_res = data_batch['res']
            label_mas = data_batch['mas']

            pred_result = model(encoder_input, encoder_input_mask, image_size, transcripts, \
                encoder_input_bboxes, extractor, tokenizer, bert
            )

            # post-procrssing
            ly_cls_preds, (re_cls_preds, ma_att_preds) = pred_result
            
            if re_cls_preds:
                re_cls_preds = torch.stack(re_cls_preds, dim=1)
                ma_att_preds = torch.cat(ma_att_preds, dim=1)
                for batch_idx in range(len(image_size)):
                    ly_cls_preds_pb = ly_cls_preds[batch_idx]
                    re_cls_preds_pb = re_cls_preds[batch_idx]
                    ma_att_preds_pb = ma_att_preds[batch_idx]
                    ly_label = batch_label_lys[batch_idx]
                    ly_p, ly_r, ly_f1 = ly_prf(ly_cls_preds_pb, ly_label)
                    label_ma, label_re = label_mas[batch_idx], label_res[batch_idx]
                    p, r, f1 = cal_f_measure(ly_cls_preds_pb, ly_label, re_cls_preds_pb, ma_att_preds_pb, label_re, label_ma)
                    p_lst.append(p), r_lst.append(r), f_lst.append(f1)
                    lines = batch_lines[batch_idx]
                    pdf_path = pdf_paths[batch_idx]
                    pred_tree = post_process_visual(pdf_path, lines, ly_cls_preds_pb, cfg.ly_vocab, cfg.re_vocab, re_cls_preds_pb, ma_att_preds_pb, stride=0.8)
                    label_tree = batch_label_synatx_tree[batch_idx]
                    tree_score = tree_edit_distance(pred_tree, label_tree)
                    result_temp = {'TEDS':tree_score,'heading_detection_p':ly_p, 'heading_detection_p_r':ly_r,'toc_p':p, 'toc_r':r, 
                                    'predict_tree':repr(pred_tree).split('\n'), 'label_tree':repr(label_tree).split('\n')}
                    all_predicted_result.append(result_temp)
            else:
                print('no title predicted in ', pdf_paths)
            # break # debug
        except RuntimeError as E:
            if 'out of memory' in str(E):
                logger.info('iter = ' + str(it) + ' CUDA Out Of Memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info('iter = ' + str(it) + ' ' + str(E))

    total_teds = sum([r['TEDS'] for r in all_predicted_result]) / len(all_predicted_result)
    total_p = sum([r['ly_p'] for r in all_predicted_result]) / len(all_predicted_result)
    total_r = sum([r['ly_r'] for r in all_predicted_result]) / len(all_predicted_result)
    total_f = 2 / (1 / total_p + 1 / total_r)
    logger.info('TEDS={}, ly_f1={}, ly_p={}, ly_r={}'.format(total_teds, total_f, total_p, total_r))
    logger.info('Heading Detection: F1 = {}, P = {}, R = {}'.format(total_f, total_p, total_r))
    avg_p, avg_r = sum(p_lst) / len(p_lst), sum(r_lst)/len(r_lst)
    avg_f = 2 / (1 / avg_p + 1 / avg_r)
    logger.info('ToC Extraction: TEDS = {}, P = {}, R = {}, F1 = {}'.format(total_teds, avg_p, avg_r, avg_f))
    with open(os.path.join(cfg.work_dir, 'infer.json'), 'w') as f:
        json_data = json.dumps(all_predicted_result, indent=4)
        f.write(json_data)
    return total_teds, total_f





def main():
    init()

    valid_dataloader = create_valid_dataloader(cfg.ly_vocab, cfg.re_vocab, cfg.valid_pickle_path, cfg.valid_batch_size, cfg.valid_num_workers, cfg.all_labels_path)
    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(valid_dataloader.dataset),
                len(valid_dataloader.batch_sampler),
                valid_dataloader.batch_size
            )
    )

    model = build_model(cfg)
    model.cuda()

    if cfg.eval_checkpoint is not None:
        load_checkpoint(cfg.eval_checkpoint, model)
        logger.info('load checkpoint from: %s' % cfg.eval_checkpoint)
        
    with torch.no_grad():
        valid(cfg, valid_dataloader, model, 0)


if __name__ == '__main__':
    def setup_seed(seed):
        import torch, numpy as np, random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    setup_seed(2021)
    main()
