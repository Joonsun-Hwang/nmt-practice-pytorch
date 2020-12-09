import warnings
import os
import argparse
import random
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS
from torch.cuda.amp import autocast, GradScaler
import torch_optimizer as optim

from datasets import WMTDatasets
from models.models import Transformer
from utils import cal_performance, setup, cleanup

warnings.filterwarnings(action="ignore")

def patch_src(args, src, seq_len):
    if args.batch_size != src.size(0):  # if src is ([max_len, batch_size])
        src = src.transpose(0, 1)
    src = src[:, :seq_len+2]
    return src  # ([batch_size, seq_len])

def patch_trg(args, trg, seq_len):
    if args.batch_size != trg.size(0):  # if trg is ([max_len, batch_size])
        trg = trg.transpose(0, 1)
    trg, gold = trg[:, :seq_len+1], trg[:, 1:seq_len+2].contiguous().view(-1)
    return trg, gold  # ([batch_size, seq_len-1]), ([batch_size*(seq_len-1)])

def train_epoch(args, iterator, model, optimizer):
    
    model.train()
    epoch_loss, n_word_total, n_word_correct = 0, 0, 0

    for batch in tqdm(iterator, desc='  - (Training)   ', leave=False, mininterval=1):
        
        # Prepare data
        src_tensor = batch[0]
        trg_tensor = batch[1]
        src_len = batch[2]
        trg_len = batch[3]

        src = patch_src(args, src_tensor, torch.max(src_len)).to(args.rank)
        trg, gold = map(lambda x: x.to(args.rank), 
            patch_trg(args, trg_tensor, torch.max(trg_len)))
        
        # Forward
        optimizer.zero_grad()
        pred = model(src, trg)
        loss, n_correct, n_word = cal_performance(
            pred, gold, args.trg_pad_idx)

        # Backward & update parameters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 
            args.grad_clip)
        optimizer.step()
        
        # Cumulate perform
        epoch_loss += loss.item()
        n_word_total += n_word
        n_word_correct += n_correct

    avg_loss = epoch_loss / n_word_total
    acc = n_word_correct / n_word_total
    
    return avg_loss, acc
    
def val_epoch(args, iterator, model):

    model.eval()
    epoch_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc='  - (Validation) ', leave=False, mininterval=1):
        
            # Prepare data
            src_tensor = batch[0]
            trg_tensor = batch[1]
            src_len = batch[2]
            trg_len = batch[3]

            src = patch_src(args, src_tensor, torch.max(src_len)).to(args.rank)
            trg, gold = map(lambda x: x.to(args.rank), 
                patch_trg(args, trg_tensor, torch.max(trg_len)))
            
            # Forward
            pred = model(src, trg)
            loss, n_word, n_correct = cal_performance(
                pred, gold, args.trg_pad_idx)

            # Cumulate perform
            epoch_loss += loss.item()
            n_word_total += n_word
            n_word_correct += n_correct

    avg_loss = epoch_loss / n_word_total
    acc = n_word_correct / n_word_total
    
    return avg_loss, acc

def train(rank, args):
    args.rank = rank
    
    torch.cuda.set_device(args.rank)
    setup(args.rank, args.world_size)
    
    # Prepare dataset
    train_dataset = WMTDatasets(args, split='train')
    val_dataset = WMTDatasets(args, split='val')
    
    train_sampler = DS(train_dataset, num_replicas=args.world_size, rank=args.rank)
    val_sampler = DS(val_dataset, num_replicas=args.world_size, rank=args.rank)
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    
    args.src_vocab_size = train_dataset.get_src_vocab_size()
    args.trg_vocab_size = train_dataset.get_trg_vocab_size()
    args.src_pad_idx = train_dataset.get_src_pad_idx()
    args.trg_pad_idx = train_dataset.get_trg_pad_idx()
    args.max_len = train_dataset.get_max_len()

    # Initialize model & optimizer
    model = Transformer(
        n_src_vocab=args.src_vocab_size,
        n_trg_vocab=args.trg_vocab_size,
        src_pad_idx=args.src_pad_idx,
        trg_pad_idx=args.trg_pad_idx,
        d_model=args.d_model,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout,
        prj_share_weight=args.prj_share_weight,
        emb_share_weight=args.emb_share_weight).cuda(args.rank)
    optimizer = optim.RAdam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate)
        
    model = DDP(model, device_ids=[args.rank])

    # Train the model
    args.start_epoch = 0
    
    if args.rank == 0:
        print(f"[*] Start training the model with multi-GPUs {args.gpu_idxs}.")
    
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.rank == 0:
            print("\nepoch: [" + str(epoch) + "/" + str(args.max_epoch) + "]")
        
        # training phase
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            args=args,
            iterator=train_data_loader,
            model=model,
            optimizer=optimizer)
        print_performances('Training', train_loss, train_acc, start_time)
        
        # validation phase
        start_time = time.time()
        val_loss, val_acc = evaluate_epoch(
            args=args,
            iterator=val_data_loader,
            model=model)
        print_performances('Validation', val_loss, val_acc, start_time)
        
        args.val_losses += [val_loss]
        args.val_acces += [val_acc]
        
        args.start_epoch = epoch+1
        if rank == args.start_rank_idx:
            save_checkpoint(args, model.state_dict(), optimizer.state_dict())
        
    if rank == 0:
        print('\n[*] End training the model.')
        print(' - minimum validation ppl: {ppl: 3.3f}, '  \
              'maximum validation accuracy: {acc: 3.3f}'.format(
                  ppl=math.exp(min(args.val_losses, 100)), acc=100*max(args.val_acces)))
    
    cleanup()
    

if __name__ == "__main__":
    '''
    CUDA_VISIBLE_DEVICES=1,2,3 python train.py transformer --emb_share_weight --prj_share_weight --batch_size 1000
    '''
    # Operating System Setup
    try:
        gpu_idxs = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
    except KeyError:
        gpu_idxs = [0]
    
    # Prepare parser
    parser = argparse.ArgumentParser(description="Training neural machine translation model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
        help="The name of directory where the datasets are saved.")
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_inner_hid', type=int, default=2048)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--emb_share_weight', action='store_true')
    parser.add_argument('--prj_share_weight', action='store_true')

    # Training arguments
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1024,
        help="The number of examples in each batch")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('checkpoint', type=str,
                        help='The path of checkpoint where the trained model will be saved')
    
    # Parallel arguments
    parser.add_argument('--nodes', type=int, default=1)
    
    args = parser.parse_args()

    # Additional arguments for Multi-gpus
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpus = torch.cuda.device_count()
    args.gpu_idxs = gpu_idxs
    args.world_size = args.n_gpus * args.nodes
    args.batch_size = int(args.batch_size / args.world_size)
    args.start_rank_idx = gpu_idxs[0]
    
    # Additional arguments for Training
    args.val_losses = []
    args.val_acces = []
    args.start_epoch = 0
    
    assert args.n_gpus == len(args.gpu_idxs)

    # For Reproducibility
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Start Training
    mp.spawn(train, 
        args=(args,), 
        nprocs=args.world_size, 
        join=True)
