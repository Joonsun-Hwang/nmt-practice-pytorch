import warnings
import os
import argparse
import random
import time

from tqdm import tqdm
import numpy as np
import torch
import torch_optimizer as optim

from datasets import WMTDatasets
from models.models import Transformer
from utils import cal_performance

warnings.filterwarnings(action="ignore")

def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src  # ([batch_size, seq_len])

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold  # ([batch_size, seq_len-1]), ([batch_size*(seq_len-1)])

def train_epoch(args, iterator, model, optimizer):
    
    model.train()
    epoch_loss, n_word_total, n_word_correct = 0, 0, 0

    for batch in tqdm(iterator, desc='\t- Training -', leave=False, mininterval=1):

        # Prepare data
        src = patch_src(batch.src[0], args.src_pad_idx).to(args.device)
        trg, gold = map(lambda x: x.to(args.device), 
            patch_trg(batch.trg[0], args.trg_pad_idx))
        
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
        for batch in tqdm(iterator):

            # Prepare data
            src = patch_src(batch.src[0], args.src_pad_idx).to(args.device)
            trg, gold = map(lambda x: x.to(args.device), 
                patch_trg(batch.trg[0], args.trg_pad_idx))
            
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

def train(args):
    
    # Prepare dataset
    train_dataset = WMTDatasets(args, split='train')
    val_dataset = WMTDatasets(args, split='val')
    
    train_iterator = train_dataset.get_data_iterator()
    val_tierator = val_dataset.get_data_iterator()
    
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
        emb_share_weight=args.emb_share_weight).to(args.device)
    optimizer = optim.RAdam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate)

    # Train the model
    val_losses = []
    val_acces = []
    start_epoch = 0
    
    print("[*] Start training the model.")
    for epoch in range(start_epoch, args.max_epoch):
        print("\nepoch: [" + str(epoch) + "/" + str(args.max_epoch) + "]")
        
        # training phase
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            args=args,
            iterator=train_iterator,
            model=model,
            optimizer=optimizer)
        print_performances('Training', train_loss, train_acc, start_time)
        
        # validation phase
        start_time = time.time()
        val_loss, val_acc = evaluate_epoch(
            args=args,
            iterator=val_tierator,
            model=model)
        print_performances('Validation', val_loss, val_acc, start_time)
        
        val_losses += [val_loss]
        val_acces += [val_acc]
        
        args.start_epoch = epoch+1
        save_checkpoint(args, model.state_dict())
    
    print('\n[*] End training the model.')
    print(' - minimum validation ppl: {ppl: 3.3f}, '  \
          'maximum validation accuracy: {acc: 3.3f}'.format(
              ppl=math.exp(min(loss, 100)), acc=100*max(val_acces)))
    

if __name__ == "__main__":
    '''
    python train.py transformer --emb_share_weight --prj_share_weight
    '''
    
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
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024,
        help="The number of examples in each batch")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('checkpoint', type=str,
                        help='The path of checkpoint where the trained model will be saved')
    
    args = parser.parse_args()
    
    # Additional arguments
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train(args)
