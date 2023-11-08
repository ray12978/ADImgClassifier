import logging
import os
import re


def get_logger(name='', fp='', exp_name='exp'):
    """Sets up logging for the given name."""
    # 1.setup log path and create log directory
    logName = f'{exp_name}.log'
    log_path = os.path.join(fp, logName)
    # create log directory
    os.makedirs(fp, exist_ok=True)
    level = logging.INFO
    # 2.create logger, then setLevel
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 3.create file handler, then setLevel
    # create file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # 4.create stram handler, then setLevel
    # create stream handler
    streamandler = logging.StreamHandler()
    streamandler.setLevel(level)
    # 5.create formatter, then handler setFormatter
    AllFormatter = logging.Formatter("%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(AllFormatter)
    streamandler.setFormatter(AllFormatter)
    # 6.logger addHandler
    logger.addHandler(file_handler)
    logger.addHandler(streamandler)
    return logger


def get_result_subdir(model_names, result_path='Results', exp_name=''):
    num = 1
    res = []
    os.makedirs(result_path, exist_ok=True)
    while os.path.exists(os.path.join(result_path, f'{exp_name}_{num}')):
        num += 1
    folder_dir = os.path.join(result_path, f'{exp_name}_{num}')
    for name in model_names:
        name = re.sub(r'hf-hub:', '', name)
        result_dir = os.path.join(folder_dir, name)
        res.append(result_dir)
    return res, folder_dir
