import torch
import argparse
import os
from os import path as ospath
from time import time, sleep
import numpy as np
from pandas import read_csv as pdreader
import pandas as pd
import warnings

def setup_environment(args):
    """
    Set up the environment based on the provided arguments.

    Args:
        args: The input arguments for setting up the environment.

    Raises:
        RuntimeError: If CUDA is not available on the system.
    """
    # if not args.disable_wandb:
    #     _wandb_data_dir = f"{args.log_path}/wandb_cache" if args.log_path else f"{os.path.expanduser('~')}/wandb_cache"
    #     os.makedirs(_wandb_data_dir, exist_ok=True)
    #     os.environ['WANDB_DATA_DIR'] = _wandb_data_dir
    #     os.environ['WANDB_CACHE_DIR'] = _wandb_data_dir

    gpu_ok = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    else:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (8, 6) ,(9, 0)):
            gpu_ok = True
    
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, A6000 or H100. Taining may be slower than excpected."
        )
    
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Set device to GPU if CUDA is available
    args.device = torch.device("cuda")

    # Set random seeds for reproducibility
    if hasattr(args, 'seed') and args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # set checkpoint path
    args.checkpoint_path = None
    if args.checkpoint_id:
        args.checkpoint_path = '/'.join([args.log_path, args.checkpoint_id])

def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)

def _load_data(fpath):
    if fpath.endswith('.npz'):
        tmp_data = np.load(fpath)
        colnames = list(tmp_data.dtype.names)
        return tmp_data, colnames
    else:
        tmp_data = pdreader(fpath, sep='\t' if fpath.endswith('.tsv') else ',').to_numpy()
    
    # extract colnames from original data
    with open(fpath, 'r') as file:
        first_line = file.readline().strip()
    colnames = first_line.split('\t' if fpath.endswith('.tsv') else ',')

    return tmp_data, colnames

def _rescale(data, eps=1e-6):
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)

    data_scaled = (data - mins) / (maxs - mins + eps)

    return data_scaled, mins, maxs

def _getscales(data, eps=1e-6):
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)

    # data_scaled = (data - mins) / (maxs - mins + eps)

    return mins, maxs

def _get_ctxembed_size(fname):
    tmp_embd = np.load(fname)
    return tmp_embd['colembed'].shape[1]


def load_eval_data(fpath, eps=1e-6):
    raw_data, colnames = _load_data(fpath)

    print(f"Loaded eval dataset of shape: {raw_data.shape} ")

    return dict(
        data = raw_data,
        colnames = colnames,
        fname = fpath.strip().split('/')[-1]
    )

def save_predictions(predictions, cnames, spath, sname, index=None):
    # make outdir if not exists
    os.makedirs(spath, exist_ok = True) 

    results = pd.DataFrame(predictions, columns=cnames)
    fnparts = sname.strip().split('.')[0]
    fnparts = fnparts.strip().split('-')

    ofname =  f"{spath}/{fnparts[0]}-predicted{'-'+'-'.join(fnparts[1:]) if len(fnparts)>1 else ''}.tsv"
    
    results.to_csv(ofname, sep='\t', index=False)

    print('Predictions saved to:', f'{ofname}')

def train_test_split_indices(total_samples, test_proportion):
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        test_size = int(total_samples * test_proportion)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        return train_indices, test_indices