import torch
import argparse
import os
from os import path as ospath
from time import time, sleep
import numpy as np
from pandas import read_csv as pdreader
import pandas as pd
from src.utils.training_helpers import setup_environment, _get_ctxembed_size, _load_data, _rescale, _getscales, load_eval_data, save_predictions

def args_parser():
    """
    Parses the custom arguments from the command line and returns the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--project', type=str, default=None, required=False)
    parser.add_argument('--tabular', type=str, default=None)
    parser.add_argument('--tabularcm', type=str, default=None)
    parser.add_argument('--tabular_infer', type=str, default=None)
    parser.add_argument('--finfer', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=None)
    parser.add_argument('--splits', type=str, required=False, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--enable_wandb', action='store_true', default=False) 
    parser.add_argument('--table_size', type=int, default=372)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--checkpoint_id', type=str, default=None)
    parser.add_argument('--binary_map', type=str, default=None)
    parser.add_argument('--context_size', type=int, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--nencoder', type=int, default=6)
    parser.add_argument('--ndecoder', type=int, default=4)
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--cembed_size', type=int, default=None)
    parser.add_argument('--loss_type', type=str, default=None)
    parser.add_argument('--ctx_prop', type=float, default=None)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    description = f'{args.model}'
    for conf in ['tabular', 'tabularcm']: # , 'labels'
        opt = getattr(args, conf)
        if opt is not None:
            opt = opt.strip('/')
            opt = opt.split('/')[-1]
            opt = opt.split('.')[0]
            description += f'-{conf}_{opt}'
    if args.project:
        print('Project:', args.project + ' :: ' + description)
    else:
        print('Project:', description)
    args.description = description

    return args

def load_data(args):
    modalities = [args.tabular, args.tabularcm]
    data_paths = [mod for mod in modalities if mod is not None]
    assert len(data_paths)
    data_path = data_paths[0]

    # Get and Set colembed dim
    if args.embeddings is not None and args.cembed_size is None:
       args.cembed_size =  _get_ctxembed_size(args.embeddings)
       print(f"Automatically set args.cembed_size to {args.cembed_size}")

    if args.tabular:
        raw_data, colnames = _load_data(args.tabular)

        print(f"Loaded dataset of shape: {raw_data.shape} ")
        samples = raw_data.shape[0]
        features = raw_data.shape[1]

        # data, minscale, maxscale = _rescale(raw_data)
        minscale, maxscale = _getscales(raw_data)

        return dict(
            data = raw_data,
            minscale = minscale,
            maxscale = maxscale,
            colnames = colnames,
            fname = args.tabular.strip().split('/')[-1]
        )

    elif args.tabularcm:
        raise NotImplementedError

    else:
        raise NotImplementedError


def load_model(args, data_dict):
    if args.model == 'RMAE': 
        from src.imputers.randmae import RandomMAE 
        if args.mask_ratio is None:
            raise ValueError('mask_ratio is required to run ReMasker!')
        
        model = RandomMAE(
            args,
            feats = data_dict['colnames']
        )
    
    elif args.model == 'CMAE':
        from src.imputers.copymae import CopyMAE
        if args.mask_ratio is None:
            raise ValueError('mask_ratio is required to run CMAE!')
        
        model = CopyMAE(
            args,
            feats = data_dict['colnames']
        )

    elif args.model == 'RCTXMAE':
        from src.imputers.randctxmae import RandomContextMAE
        if args.mask_ratio is None:
            raise ValueError('mask_ratio is required to run RCTXMAE!')
        if args.embeddings is None:
            raise ValueError('embeddings are required to run RCTXMAE!')
        
        model = RandomContextMAE(
            args,
            feats = data_dict['colnames']
        )

    elif args.model == 'CACTIabl':
        from src.imputers.cacti_abl import CACTIabl
        if args.mask_ratio is None:
            raise ValueError('mask_ratio is required to run CACTI!')
        if args.embeddings is None:
            raise ValueError('embeddings are required to run CACTI!')
        if args.loss_type is None:
            raise ValueError('Loss Type are required to run CACTI-lossctx!')
        if args.ctx_prop is None or args.ctx_prop > 1 or args.ctx_prop <= 0:
            raise ValueError('A Valid CTX proportion is required to run CACTI-lossctx!')

        model = CACTIabl(
            args,
            feats = data_dict['colnames']
        ) 
    
    elif args.model == 'CACTI':
        from src.imputers.cacti import CACTI
        if args.mask_ratio is None:
            raise ValueError('mask_ratio is required to run CACTI!')
        if args.embeddings is None:
            raise ValueError('embeddings are required to run CACTI!')
        
        model = CACTI(
            args,
            feats = data_dict['colnames']
        ) 
    
    else:
        raise NotImplementedError

    return model

def train_model(model, data_dict):
    torch.set_float32_matmul_precision('medium')

    _ = model.train_model(data_dict)

    return model

def train_eval(model, data_dict):
    torch.set_float32_matmul_precision('medium')

    # train_imputed = model.fit_transform(data_dict["data"].copy())
    train_imputed = model.fit_transform(data_dict)

    return model, train_imputed

if __name__ == "__main__":
    args = args_parser()

    setup_environment(args)

    data_dict = load_data(args)

    model = load_model(args, data_dict)

    # TRIAN only 
    if args.train_only and args.resume_checkpoint is None:
        Model = train_model(model, data_dict)
    elif args.train_only and args.resume_checkpoint:
        # FIXME: enable resume for ckpt
        raise NotImplementedError
        # trainer = resume_training(args, model, loaders)

    # TRAIN and INFER
    if args.resume_checkpoint is None:
        Model, imputed_data = train_eval(model, data_dict)
    elif args.resume_checkpoint:
        # FIXME: enable resume for ckpt
        raise NotImplementedError
        # trainer = resume_training(args, model, loaders)
    else:
        ValueError("---No checkpoint to resume---")

    # SAVE train data imputed values
    if args.save_path:
        save_predictions(imputed_data, data_dict['colnames'], 
            args.save_path, data_dict['fname'])


    # SAVE other (eg. val) data imputed values
    ### FIXME: Clean this up and make it more modular
    infer_modalities = [args.tabular_infer]
    infer_data_paths = [mod for mod in infer_modalities if mod is not None]
    if len(infer_data_paths) > 0:
        # load eval data
        data_eval_dict = load_eval_data(infer_data_paths[0])

        imputed_eval = Model.transform(data_eval_dict['data'])

        if args.save_path:
            save_predictions(imputed_eval, data_eval_dict['colnames'], 
                args.save_path, data_eval_dict['fname'])
    ###


    print('---Done!---')