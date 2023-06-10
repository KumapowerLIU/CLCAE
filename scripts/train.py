"""
This file runs the main training/val loop
"""
import json
import sys
import pprint
import random
import numpy as np
import torch
import os
sys.path.append(".")
sys.path.append("..")
from options.train_options import TrainOptions
from training.contrastive_coach import ContrastiveCoach
from training.inversion_coach import InversionCoach

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    opts = TrainOptions().parse()
    if opts.use_ddp:
        rank = int(os.environ["LOCAL_RANK"])

        init_seeds(seed=1 + rank)
    else:
        init_seeds(seed=0)
    if os.path.exists(opts.exp_dir):
        print('Oops... {} already exists'.format(opts.exp_dir))
    else:
        os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)
    if opts.train_inversion:
        coach = InversionCoach(opts)
    elif opts.train_contrastive:
        coach = ContrastiveCoach(opts)
    else:
        raise ValueError ('Please select the correct model type')
    coach.train()


if __name__ == '__main__':
    main()
