import sys
sys.path.append('./')
sys.path.append('../')
import os
import json
import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_train_dataloader, create_valid_dataloader
from libs.utils import logger
from libs.utils.counter import Counter
from libs.utils.utils import cal_mean_lr
from libs.utils.checkpoint import load_checkpoint, save_checkpoint
from libs.utils.time_counter import TimeCounter
from libs.utils.comm import distributed, synchronize
from libs.utils.model_synchronizer import ModelSynchronizer
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


def train(cfg, epoch, dataloader, model, optimizer, scheduler, time_counter, synchronizer=None):
    model.train()
    tokenizer = cfg.tokenizer
    extractor = cfg.extractor.to(cfg.device)
    bert = cfg.bert.to(cfg.device)
    counter = Counter(cache_nums=cfg.cache_nums)
    for it, data_batch in enumerate(dataloader):
        encoder_input = [data.to(cfg.device) for data in data_batch['encoder_input']]
        encoder_input_mask = [data.to(cfg.device) for data in data_batch['encoder_input_mask']]
        encoder_input_bboxes = [[torch.tensor(page).to(cfg.device).float() for page in data] for data in data_batch['bboxes']]
        transcripts = data_batch['transcripts']
        image_size = data_batch['image_size']
        ly = data_batch['ly'].to(cfg.device)
        ly_mask = data_batch['ly_mask'].to(cfg.device)
        re = data_batch['re'].to(cfg.device)
        re_mask = data_batch['re_mask'].to(cfg.device)
        ma = data_batch['ma'].to(cfg.device)
        ma_mask = data_batch['ma_mask'].to(cfg.device)

        try:
            optimizer.zero_grad()
            result_info = model(encoder_input, encoder_input_mask, image_size, transcripts, \
                encoder_input_bboxes, extractor, tokenizer, bert, ly, ly_mask, re, re_mask, ma, ma_mask
            )
            loss = sum([val for key, val in result_info.items() if 'loss' in key])
            loss.backward()
            optimizer.step()
            scheduler.step()

            counter.update(result_info)

        except RuntimeError as E:
                if 'out of memory' in str(E):
                    logger.info('CUDA Out Of Memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(E))

        if it % cfg.log_sep == 0:
            logger.info(
                '[Train][Epoch %03d Iter %04d][Memory: %.0f][Mean LR: %f][Left: %s] %s' % \
                (
                    epoch,
                    it,
                    torch.cuda.max_memory_allocated()/1024/1024,
                    cal_mean_lr(optimizer),
                    time_counter.step(epoch, it + 1),
                    counter.format_mean(sync=False)
                )
            )

        if synchronizer is not None:
            synchronizer()
        
        scheduler.step()
            
    if synchronizer is not None:
        synchronizer(final_align=True)


def valid(cfg, dataloader, model, epoch):
    model.eval()
    tokenizer = cfg.tokenizer
    extractor = cfg.extractor.to(cfg.device)
    bert = cfg.bert.to(cfg.device)
    all_predicted_result = list()
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
                    lines = batch_lines[batch_idx]
                    pdf_path = pdf_paths[batch_idx]
                    pred_tree = post_process_visual(pdf_path, lines, ly_cls_preds_pb, cfg.ly_vocab, cfg.re_vocab, re_cls_preds_pb, ma_att_preds_pb, stride=0.8)
                    label_tree = batch_label_synatx_tree[batch_idx]
                    tree_score = tree_edit_distance(pred_tree, label_tree)
                    result_temp = {'TEDS':tree_score,'ly_p':ly_p, 'ly_r':ly_r, 'ly_f1':ly_f1, 'contain_f1':0, 'equal_f1':0, 'sibling_f1':0,
                                    'predict_tree':repr(pred_tree).split('\n'), 'label_tree':repr(label_tree).split('\n')}
                    all_predicted_result.append(result_temp)
            else:
                logger.info('no title predicted in ', pdf_paths)
        except RuntimeError as E:
            if 'out of memory' in str(E):
                logger.info('iter = ' + str(it) + ' CUDA Out Of Memory') # to delete
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info('iter = ' + str(it) + ' ' + str(E)) # to delete

    total_teds = sum([r['TEDS'] for r in all_predicted_result]) / len(all_predicted_result)
    total_p = sum([r['ly_p'] for r in all_predicted_result]) / len(all_predicted_result)
    total_r = sum([r['ly_r'] for r in all_predicted_result]) / len(all_predicted_result)
    total_f = sum([r['ly_f1'] for r in all_predicted_result]) / len(all_predicted_result)
    logger.info('TEDS={}, ly_f1={}, ly_p={}, ly_r={}'.format(total_teds, total_f, total_p, total_r))
    with open(os.path.join(cfg.work_dir, 'pred_result_epoch{}.json'.format(epoch)), 'w') as f:
        json_data = json.dumps(all_predicted_result, indent=4)
        f.write(json_data)
    return total_teds, total_f


def build_optimizer(cfg, model):
    params = list()
    for _, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    optimizer = torch.optim.Adam(params, cfg.base_lr)
    return optimizer


def build_scheduler(cfg, optimizer, epoch_iters, start_epoch=0):
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.num_epochs * epoch_iters,
        eta_min=cfg.min_lr,
        last_epoch=-1 if start_epoch == 0 else start_epoch * epoch_iters
    )
    return scheduler


def main():
    init()

    train_dataloader = create_train_dataloader(cfg.ly_vocab, cfg.re_vocab, cfg.train_pickle_path, cfg.train_batch_size, cfg.train_num_workers, cfg.all_labels_path)
    logger.info(
        'Train dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(train_dataloader.dataset),
                len(train_dataloader.batch_sampler),
                cfg.train_batch_size
            )
    )

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

    if distributed():
        synchronizer = ModelSynchronizer(model, cfg.sync_rate)
    else:
        synchronizer = None
    
    epoch_iters = len(train_dataloader.batch_sampler)
    optimizer = build_optimizer(cfg, model)

    metrics_name = ['TEDS', 'ly_f']
    best_metrics = [0, 0]
    start_epoch = 0
    
    resume_path = os.path.join(cfg.work_dir, 'latest_model.pth')
    if os.path.exists(resume_path):
        best_metrics, start_epoch = load_checkpoint(resume_path, model, optimizer)
        start_epoch += 1
        logger.info('resume from: %s' % resume_path)
    elif cfg.train_checkpoint is not None:
        load_checkpoint(cfg.train_checkpoint, model)
        logger.info('load checkpoint from: %s' % cfg.train_checkpoint)

    scheduler = build_scheduler(cfg, optimizer, epoch_iters, start_epoch)
    
    time_counter = TimeCounter(start_epoch, cfg.num_epochs, epoch_iters)
    time_counter.reset()

    for epoch in range(start_epoch, cfg.num_epochs):
        train(cfg, epoch, train_dataloader, model, optimizer, scheduler, time_counter, synchronizer)

        with torch.no_grad():
            metrics = valid(cfg, valid_dataloader, model, epoch)

            for metric_idx in range(len(metrics_name)):
                if metrics[metric_idx] > best_metrics[metric_idx]:
                    best_metrics[metric_idx] = metrics[metric_idx]
                    save_checkpoint(os.path.join(cfg.work_dir, 'best_%s_model.pth' % metrics_name[metric_idx]), model, optimizer, best_metrics, epoch)
                    logger.info('Save current model as best_%s_model' % metrics_name[metric_idx])

        save_checkpoint(os.path.join(cfg.work_dir, 'latest_model.pth'), model, optimizer, best_metrics, epoch)


if __name__ == '__main__':
    def setup_seed(seed):
        import torch, numpy as np, random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    # setup_seed(2021)
    setup_seed(233)
    main()
