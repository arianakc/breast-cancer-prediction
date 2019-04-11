# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-04-10 23:58
import logging
import os


def init_logger(root_dir='.', name="train.log"):
    """
    Initialize a logger
    :param root_dir: directory for saving log
    :param name: logger name
    :return: a logger
    """
    os.makedirs(root_dir, exist_ok=True)
    log_formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s| %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(os.path.join(root_dir, name))
    file_handler = logging.FileHandler(os.path.join(root_dir, name), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger
