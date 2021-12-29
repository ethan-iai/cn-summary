# -*- coding: utf-8 -*-
import os
import time
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from .data import get_dataloader
from .model import BertForSeq2Seq, set_device

from cn_summary.utils import get_src_path

# N_EPOCHS = 30
# LR = 5e-4
# WARMUP_PROPORTION = 0.1
# MAX_GRAD_NORM = 1.0
MODEL_PATH = get_src_path('./bert-base-chinese')
LOG_PATH = get_src_path('./logs')
# SAVE_DIR = get_src_path('./saved_models')

logging.basicConfig(
    filename=os.path.join(LOG_PATH, time.strftime(
        'bert-%y-%m-%d-%H-%M.out', 
        time.localtime()
    )),
    filemode='a',
    format='%(asctime)s, %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='PyTorch Bert seq2seq Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--save-dir', type=str, default=get_src_path('./saved_models'))
    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    
    return parser.parse_args()
    

def run(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_device(device)
     
    logging.info(f'Run on device: {device}')
    
    train_loader = get_dataloader(
        os.path.join(args.data, 'train.json'), 
        batch_size=args.batch_size, 
        num_workers=args.workers
    )

    val_loader = get_dataloader(
        os.path.join(args.data, 'test.json'), 
        batch_size=args.batch_size,
        num_workers=args.workers 
    )

    best_valid_loss = float('inf')
    if len(args.resume) == 0:
        model = BertForSeq2Seq.from_pretrained(MODEL_PATH)
    else:
        model = BertForSeq2Seq.from_pretrained(args.resume)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]

    steps_per_epoch = len(train_loader)
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_epochs * steps_per_epoch), 
        num_training_steps=total_steps
    )
    
    loss_vals = []
    loss_vals_eval = []
    for epoch in range(1, 1 + args.epochs):
        model.train()
        epoch_loss = []
        pbar = tqdm(train_loader)
        pbar.set_description("[Train Epoch {}]".format(epoch)) 
    
        for batch_idx, batch_data in enumerate(pbar):
            
            input_ids = batch_data["input_ids"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            token_type_ids_for_mask = batch_data["token_type_ids_for_mask"].to(device)
            labels = batch_data["labels"].to(device)
                       
            model.zero_grad()
            predictions, loss = model(input_ids, token_type_ids, token_type_ids_for_mask, labels)           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
        loss_vals.append(np.mean(epoch_loss))
        logging.info(f'[Train Epoch {epoch}] Train Loss: {loss_vals[-1]}')
        
        model.eval()
        epoch_loss_eval= []
        pbar = tqdm(val_loader)
        pbar.set_description("[Eval Epoch {}]".format(epoch))
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                input_ids = batch_data["input_ids"].to(device)
                token_type_ids = batch_data["token_type_ids"].to(device)
                token_type_ids_for_mask = batch_data["token_type_ids_for_mask"].to(device)
                labels = batch_data["labels"].to(device)
                predictions, loss = model(input_ids, token_type_ids, token_type_ids_for_mask, labels)                    
                epoch_loss_eval.append(loss.item())
                
        valid_loss = np.mean(epoch_loss_eval)
        loss_vals_eval.append(valid_loss)
        logging.info(f'[Eval Epoch {epoch}] Eval Loss: {loss_vals_eval[-1]}')    

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'bert_{epoch:03d}.bin'))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'bert_best.bin'))
        
        with torch.cuda.device(f'cuda:{args.gpu}'):
            torch.cuda.empty_cache()
   
    l1, = plt.plot(np.linspace(1, args.epochs, args.epochs).astype(int), loss_vals)
    l2, = plt.plot(np.linspace(1, args.epochs, args.epochs).astype(int), loss_vals_eval)
    plt.legend(handles=[l1, l2], labels=['Train loss','Eval loss'], loc='best')
    plt.savefig(os.path.join(args.save_dir, 'loss.png'))

if __name__ == '__main__':
    run(args=parse_args())