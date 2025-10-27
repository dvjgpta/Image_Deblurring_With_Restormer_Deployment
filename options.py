import argparse
import random
import torch

from utils import parse

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',type=str,required=True,help='Path to option Yaml file')
    parser.add_argument('--launcher',choices=['none','pytorch','slurm'],default='none',help='Job Launcher')
    parser.add_argument('--local_rank',type=int,default=0)
    args=parser.parse_args()

    opt=parse(args.opt,is_train=is_train)

    #distributed setup
    if args.launcher!='none':
        from utils import init_dist
        init_dist(args.launcher)
        opt['dist']=True
    else:
        opt['dist']=False

    
    #random seed
    seed=opt.get('manual_seed',random.randint(1,1000))
    torch.manual_seed(seed)
    random.seed(seed)
    opt['manual_seed']= seed

    return opt
#opt is a dictionary storing everything
