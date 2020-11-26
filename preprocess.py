import urllib
import os
import sys
import argparse
import tarfile
from glob import glob

import dill as pickle
from tqdm import tqdm
import torch
from torchtext import data, datasets
from torchtext.datasets import TranslationDataset
import sentencepiece as spm

from utils import file_exist, mkdir_if_needed, rmdir_if_existed, rmfile_if_existed
import special_tokens

HERE = os.path.dirname(os.path.realpath(__file__))


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize

def _download_and_extract(download_dir, url, src_filename, trg_filename):
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        sys.stderr.write(f"[-] Already downloaded and extracted {url}.\n")
        return src_path, trg_path

    compressed_file = _download_file(download_dir, url)

    sys.stderr.write(f"[*] Extracting {compressed_file}.\n")
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        corpus_tar.extractall(download_dir)

    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)
    
    if src_path and trg_path:
        return src_path, trg_path

    raise OSError(f"[!] Download/extraction failed for url {url} to path {download_dir}")

def _download_file(download_dir, url):
    filename = url.split("/")[-1]
    print(download_dir)
    if file_exist(download_dir, filename):
        sys.stderr.write(f"[-] Already downloaded: {url} (at {filename}).\n")
    else:
        sys.stderr.write(f"[*] Downloading from {url} to {filename}.\n")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename

def get_raw_files(raw_dir, sources):
    raw_files = { "src": [], "trg": [], }
    for source in sources:
        src_file, trg_file = _download_and_extract(raw_dir, source["download_url"], source["src_raw_fname"], source["trg_raw_fname"])
        raw_files["src"].append(src_file)
        raw_files["trg"].append(trg_file)
    return raw_files

def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"[-] Merged files found, skip the merging process.\n")
        src_cnt = sum(1 for line in open(src_fpath))
        return src_fpath, trg_fpath, src_cnt

    sys.stderr.write(f"[*] Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'\
                    f'    - SRC: {src_inf}\n' \
                    f'    - TRG: {trg_inf}\n')
            with open(src_inf, newline='\n') as src_inf, open(trg_inf, newline='\n') as trg_inf:
                src_cnt = trg_cnt = 0
                for i, line in enumerate(src_inf):
                    src_cnt += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    trg_cnt += 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert src_cnt == trg_cnt, '[!] Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath, src_cnt

def build_vocabulary(args):
    train_fpath = os.path.join(args.raw_dir, '-'.join(["raw", args.prefix, "train"]))
    val_fpath = os.path.join(args.raw_dir, '-'.join(["raw", args.prefix, "val"]))
    test_fpath = os.path.join(args.raw_dir, '-'.join(["raw", args.prefix, "test"]))
    sp_fname = '-'.join(["sp", args.prefix])

    if os.path.isfile(sp_fname+'.model'):
        sys.stderr.write(f"[-] Trained Sentence Piece model {sp_fname} found, skip the merging process.\n")
    else:
        sys.stderr.write(f"[*] Training Sentence Piece model: {sp_fname}...\n")
        sp_args = {
            "input": ",".join(glob(train_fpath+'*')), 
            "model_prefix": sp_fname, 
            "vocab_size": args.vocab_size, 
            "bos_id": -1,
            "eos_id": -1,
            "character_coverage": 1.0,
            "model_type": "unigram", 
            "normalization_rule_name": "nfkc_cf"}
        joined_sp_args = " ".join(
            "--{}={}".format(k, v) for k, v in sp_args.items())
        spm.SentencePieceTrainer.Train(joined_sp_args)
        
    sp = spm.SentencePieceProcessor()
    sp.load(sp_fname+'.model')
    
    sys.stderr.write(f"[*] Preparing vocabulary...\n")
    if args.split_vocab:
        SRC = data.Field(
            tokenize=sp.encode_as_pieces, lower=not args.keep_case, include_lengths=True,
            pad_token=special_tokens.PAD_PIECE, init_token=special_tokens.BOS_PIECE, eos_token=special_tokens.EOS_PIECE)
        TRG = data.Field(
            tokenize=sp.encode_as_pieces, lower=not args.keep_case, include_lengths=True,
            pad_token=special_tokens.PAD_PIECE, init_token=special_tokens.BOS_PIECE, eos_token=special_tokens.EOS_PIECE)
        fields = (SRC, TRG)
    else:
        sys.stderr.write(f"\t[-] The vocabularies are shared.\n")
        FLD = data.Field(
            tokenize=sp.encode_as_pieces, lower=not args.keep_case, include_lengths=True,
            pad_token=special_tokens.PAD_PIECE, init_token=special_tokens.BOS_PIECE, eos_token=special_tokens.EOS_PIECE)
        fields = (FLD, FLD)
    
    MAX_LEN = args.max_len
    def _filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
    
    sys.stderr.write(f"[*] Preparing dataset...\n")
    train = TranslationDataset(
        fields=fields,
        path=train_fpath,
        exts=('.src', '.trg'),
        filter_pred=_filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=val_fpath,
        exts=('.src', '.trg'),
        filter_pred=_filter_examples_with_length)
    test = TranslationDataset(
        fields=fields,
        path=test_fpath,
        exts=('.src', '.trg'),
        filter_pred=_filter_examples_with_length)
        
    sys.stderr.write(f"[*] Building the vocabularies...\n")
    if args.split_vocab:
        fields[0].build_vocab(train.src, min_freq=args.min_freq)
        fields[1].build_vocab(train.trg, min_freq=args.min_freq)
    else:
        fields[0].build_vocab(train, min_freq=args.min_freq)
    
    train_val_data = {
        'settings': args,
        'vocab': {'src': fields[0], 'trg': fields[1]},
        'train': train.examples,
        'val': val.examples}
    test_data = {
        'settings': args,
        'vocab': {'src': fields[0], 'trg': fields[1]},
        'test': test.examples}
        
    print('[*] Dumping the processed data to data directory', args.data_dir)
    pickle.dump(train_val_data, open(os.path.join(args.data_dir, 'train_val.pkl'), 'wb'))
    pickle.dump(test_data, open(os.path.join(args.data_dir, 'test.pkl'), 'wb'))
    
    print('\n\n[*] RESULTS:')
    print('\t', len(train.examples), 'training data', len(val.examples), 'validation data and', len(test.examples), 'test data were preprocessed.')
    print('\t', len(fields[0].vocab), "source vocabulary and", len(fields[1].vocab), "target vocabulary were saved.")

if __name__ == "__main__":
    """
    simple run:
    python preprocess.py
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--raw_dir', type=str, default=os.path.join(HERE, 'raw_data'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(HERE, 'data'))
    parser.add_argument('--src_lang', type=str, default='de')
    parser.add_argument('--trg_lang', type=str, default='en')
    
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=32000, 
        help="The nubmer of tokens for training")
    parser.add_argument('--min_freq', type=int, default=0, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument('--keep_case', action='store_true')
    parser.add_argument('--split_vocab', action='store_true')
    
    args = parser.parse_args()
    
    args.prefix = '-'.join([args.src_lang, args.trg_lang])
    
    _TRAIN_DATA_SOURCES = [
        {"download_url": "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz",
         "src_raw_fname": '.'.join(["news-commentary-v13", args.src_lang+'-'+args.trg_lang, args.src_lang]),
         "trg_raw_fname": '.'.join(["news-commentary-v13", args.src_lang+'-'+args.trg_lang, args.trg_lang])}
        ]
    _VAL_DATA_SOURCES = [
        {"download_url": "http://data.statmt.org/wmt18/translation-task/dev.tgz",
         "src_raw_fname": '.'.join(["newstest2013", args.src_lang]),
         "trg_raw_fname": '.'.join(["newstest2013", args.trg_lang])}
        ]
    _TEST_DATA_SOURCES = [
        {"download_url": "https://storage.googleapis.com/tf-perf-public/official_transformer/test_data/newstest2014.tgz",
         "src_raw_fname": '.'.join(["newstest2014", args.src_lang]),
         "trg_raw_fname": '.'.join(["newstest2014", args.trg_lang])}
        ]
    
    # Create folder if needed.
    mkdir_if_needed(args.raw_dir)
    mkdir_if_needed(args.data_dir)
    
    # Download and extract raw data.
    raw_train = get_raw_files(args.raw_dir, _TRAIN_DATA_SOURCES)
    raw_val = get_raw_files(args.raw_dir, _VAL_DATA_SOURCES)
    raw_test = get_raw_files(args.raw_dir, _TEST_DATA_SOURCES)

    # Merge files into one.
    train_src, train_trg, train_cnt = compile_files(args.raw_dir, raw_train, args.prefix + '-train')
    val_src, val_trg, val_cnt = compile_files(args.raw_dir, raw_val, args.prefix + '-val')
    test_src, test_trg, test_cnt = compile_files(args.raw_dir, raw_test, args.prefix + '-test')
    
    build_vocabulary(args)
    
    rmdir_if_existed(args.raw_dir)
    rmfile_if_existed('*.tgz')
    rmfile_if_existed('sp-'+args.prefix+'*')
