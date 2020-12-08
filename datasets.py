import os
import pickle

import torch
from torch.utils.data import Dataset
from torchtext import data, datasets

import special_tokens


class WMTDatasets(Dataset):
    def __init__(self, args, split):
        super(WMTDatasets, self).__init__()

        assert split in ['train', 'val', 'test'], "[!] The argument of 'split' should be 'train', 'val',  or 'test'."
        self.split = split
        self.args = args

        self.data = pickle.load(open(os.path.join(args.data_dir, split+'.pkl'), 'rb'))
        
        self.settings = self.data['settings']

        self.max_len = self.settings.max_len
        
        self.src = self.data['vocab']['src']
        self.trg = self.data['vocab']['trg']
        self.fields = {'src': self.src, 'trg': self.trg}
        
        self.examples = list(self.data['examples'])
        
    """
    def get_data_iterator(self):
        dataset = data.Dataset(examples=self.data['examples'], fields=self.fields)
        if self.split in ['train', 'val']:
            data_iterator = data.BucketIterator(dataset, 
                batch_size=self.args.batch_size, device=self.args.device, shuffle=True,
                sort_key=lambda x: len(x.trg) + (self.max_len * len(x.src)), sort_within_batch = True)
        else:
            data_iterator = data.BucketIterator(dataset,
                batch_size=1, device=self.args.device, shuffle=False, train=train)
        
        return data_iterator
   """
   
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        src_plain = self.examples[idx].__dict__['src']
        trg_plain = self.examples[idx].__dict__['trg']
        src_len = len(src_plain)
        trg_len = len(trg_plain)
        
        src_tensor = [self.src.vocab.stoi[special_tokens.BOS_PIECE]]
        for word in src_plain:
            src_tensor.append(self.src.vocab.stoi[word])
        src_tensor.append(self.src.vocab.stoi[special_tokens.EOS_PIECE])
        src_tensor = src_tensor + [self.src.vocab.stoi[special_tokens.PAD_PIECE]] * (self.max_len - src_len)
        
        trg_tensor = [self.trg.vocab.stoi[special_tokens.BOS_PIECE]]
        for word in trg_plain:
            trg_tensor.append(self.trg.vocab.stoi[word])
        trg_tensor.append(self.trg.vocab.stoi[special_tokens.EOS_PIECE])
        trg_tensor = trg_tensor + [self.trg.vocab.stoi[special_tokens.PAD_PIECE]] * (self.max_len - trg_len)
        
        return torch.LongTensor(src_tensor), torch.LongTensor(trg_tensor), \
            torch.LongTensor([src_len]), torch.LongTensor([trg_len])
    
    def __len__(self):
        return len(self.examples)

    def get_src_pad_idx(self):
        return self.src.vocab.stoi[special_tokens.PAD_PIECE]
        
    def get_trg_pad_idx(self):
        return self.trg.vocab.stoi[special_tokens.PAD_PIECE]

    def get_src_vocab_size(self):
        return len(self.src.vocab)

    def get_trg_vocab_size(self):
        return len(self.trg.vocab)
    
    def get_max_len(self):
        return self.max_len
