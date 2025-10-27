import yaml
import json
import logging
import os
from torch.utils.tensorboard import SummaryWriter
import os
import wandb
from easydict import EasyDict as edict

def parse(opt_path, is_train=True):
    """Parse YAML or JSON config file into a dictionary."""
    if opt_path.endswith('.yaml') or opt_path.endswith('.yml'):
        with open(opt_path, 'r') as f:
            opt = yaml.safe_load(f)
    elif opt_path.endswith('.json'):
        with open(opt_path, 'r') as f:
            opt = json.load(f)
    else:
        raise ValueError('Only YAML and JSON config files are supported.')

    # Convert to EasyDict for dot-access
    opt = edict(opt)

    # Add default keys if missing
    if 'logger' not in opt:
        opt.logger = edict({'use_tb_logger': True, 'wandb': None})
    if 'datasets' not in opt:
        opt.datasets = {}
    if 'train' not in opt:
        opt.train = edict({'total_iter': 100000, 'warmup_iter': -1})

    return opt



def get_root_logger(logger_name='experiment', log_level=logging.INFO, log_file=None):
    """Get a root logger that logs to console and optionally a file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger



def init_tb_logger(log_dir):
    """Initialize TensorBoard logger."""
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = SummaryWriter(log_dir)
    return tb_logger


def init_wandb_logger(opt):
    """Initialize Weights & Biases logging."""
    wandb_config = opt.logger.wandb
    project_name = wandb_config.get('project', 'restoration')
    entity = wandb_config.get('entity', None)
    name = opt.name

    wandb.init(project=project_name, entity=entity, name=name, config=dict(opt))
    return wandb
