import logging
from utils import get_root_logger, init_tb_logger, init_wandb_logger

def init_loggers(opt):
    log_file = f"{opt['path']['log']}/train_{opt['name']}.log"
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    tb_logger = None
    if opt['logger'].get('use_tb_logger', False):
        tb_logger = init_tb_logger(log_dir=f"tb_logger/{opt['name']}")

    if opt['logger'].get('wandb', None):
        init_wandb_logger(opt)

    logger.info(f"Options: {opt}")
    return logger, tb_logger
