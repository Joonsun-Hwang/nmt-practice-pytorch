import warnings
import os
import argparse

from tqdm import tqdm
import torch

from datasets import WMTDatasets

warnings.filterwarnings(action="ignore")

def train(args):
    
    # For reproducibility
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Prepare data iterator

    # Initialize model & optimizer

    # Train the model
    

if __name__ == "__main__":
    
    # Prepare parser
    parser = argparse.ArgumentParser(description="Training neural machine translation model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
        help="The name of directory where the datasets are saved.")
    
    # Model arguments
    parser.add_argument('--batch_size', type=int, default=1024,
        help="The number of examples in each batch")

    # Training arguments

    args = parser.parse_args()
    
    # Additional arguments
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Start Training
    train(args)
    

    train_datasets = WMTDatasets(args, split='train')
    train_iterator = train_datasets.get_data_iterator()

    for batch in tqdm(train_iterator, desc='train', leave=False, mininterval=1):
        print(batch)
        exit()
