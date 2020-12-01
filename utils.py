import os
import shutil
from glob import glob

import torch
import torch.nn.functional as F

def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def rmdir_if_existed(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)

def rmfile_if_existed(file_name):
    fpaths = glob(file_name)
    for fpath in fpaths:
        if os.path.exists(fpath):
            os.remove(fpath)

def cal_performance(pred, gold, trg_pad_idx, smoothing=True):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def print_performances(header, loss, acc, start_time):
    print(' - {header:12} ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, '\
          'elapse: {elapse:3.3f} min'.format(
              header=f"({header})", ppl=math.exp(min(loss, 100)),
              acc=100*acc, elapse=(time.time()-start_time)/60))

def save_checkpoint(args, model_state_dict, optimizer_state_dict):
    here = os.path.dirname(os.path.realpath(__file__))
    checkpoint = {
        'args': args, 
        'model_state_dict': model_state_dict, 
        'optimizer_state_dict': optimizer_state_dict}
    
    ckpt_dir = 'checkpoints'
    mkdir_if_needed(ckpt_dir)
    
    file_name = args.checkpoint + '.ckpt'
    torch.save(checkpoint, os.path.join(here, ckpt_dir, file_name))
    
    if args.val_losses[-1] <= min(args.val_losses):
        file_name = 'BEST_' + file_name
        torch.save(checkpoint, os.path.join(model_path, file_name))
        print('\t[!] The best checkpoint is updated.')