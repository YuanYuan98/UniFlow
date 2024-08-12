import argparse

import random

import os

import warnings

from UniFlow_model import model_select

from train import TrainLoop

import pickle
import json

import joblib

import setproctitle

import torch

from DataLoader import data_load_index_main
from utils import *

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import resource


def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")


def create_argparser():
    defaults = dict(
        patch_size = 2,
        seq_len = 24,
        data_dir="",
        weight_decay=0.0,
        batch_size=2,
        early_stop = 10,
        log_interval=5,
        device_id='0',
        machine = 'LM1',
        mask_ratio = 0.5,
        random=True,
        eval=True,
        is_block = True,
        t_patch_size = 3,
        data_norm = 1,
        size = 'middle',
        clip_grad = 0.02,
        mask_strategy = 'causal',
        mode='training',
        file_load_path = '',
        dataset = 'TaxiBJ13_48',
        his_len = 12,
        pred_len = 12,
        lr_anneal_steps = 150,
        total_epoches = 200,
        lr = 1e-3,
        pos_emb = 'SinCos',
        no_qkv_bias = 0,
        is_prompt = 1,
        is_time_emb = 1,
        batch_size_taxibj = 128,
        batch_size_nj = 256,
        batch_size_nyc = 256,
        batch_size_crowd = 256,
        prompt_content = 'none',
        emb_tuning = 0,
        used_data = 'itself',
        num_memory = 512,
        mask_ablation = '',
        data_type = 'GridGraph',
        batch_size_graph_large = 32,
        batch_size_graph_small = 64,
        spec_mlp = 0,
        few_ratio = 1.0,
        multi_patch = True,
        seed = 100, 
        few_data = '',
        finetune = 0,
        batch_ratio = 1.0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



warnings.filterwarnings('ignore', category=FutureWarning)

torch.set_num_threads(32)

def main():

    # experiment start
    th.autograd.set_detect_anomaly(True)
    args = create_argparser().parse_args()
    setproctitle.setproctitle("GPU{}".format(args.device_id))
    setup_init(args.seed)
    args.mask_ratio = args.pred_len / (args.pred_len+args.his_len)

    # save path
    if len(args.dataset.split('*'))<10:
        data_replace = args.dataset
    else:
        data_replace = len(args.dataset.split('*'))
    args.folder = 'Size_{}_Dataset_{}_PType_{}_his_{}_pred_{}_UseData_{}_{}/'.format(args.size, data_replace, args.prompt_content,  args.his_len, args.pred_len,  args.used_data, args.machine)
    model_folder = 'Exp'
    args.model_path = './experiments_all/{}/{}'.format(model_folder,args.folder)
    logdir = "./logs_all/{}/{}".format(model_folder,args.folder)
    if not os.path.exists('./experiments_all/'):
        os.mkdir('./experiments_all/')
    if not os.path.exists('./experiments_all/{}/'.format(model_folder)):
        os.mkdir('./experiments_all/{}/'.format(model_folder))
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')
    with open(args.model_path+'result_all.txt', 'w') as f:
        f.write('start training\n')
    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)
    #device = torch.device('cpu')

    # load data
    data,  train_index, test_index, val_index, args.scaler = data_load_index_main(args)
    
    # build model
    model = model_select(args=args).to(device)
    if args.multi_patch:
        print('multi_patch')
        model.init_multiple_patch()
    model.init_prompt()
    model = model.to(device)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('params: {:4f}M'.format(para * 4 / 1000 / 1000))
    print('device:',device)


    # trianing
    args.min_lr = args.lr * 0.1

    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        data=data,
        train_index = train_index,
        test_index=test_index, 
        val_index=val_index,
        device=device,
        best_rmse = 1e9,
    ).run_loop()



if __name__ == "__main__":
    main()