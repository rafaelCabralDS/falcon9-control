import os
from typing import Dict
import yaml
from .logger import *

def show_dict(data: Dict, depth=0):
    """
    print the dictionary of configurations
    params:
        config: configurations of each variable
    """
    empty_space = ' ' * depth * 10
    if depth == 0:
        logger.info(colorize('-' * 80, color='blue', bold=True))

    for k, v in data.items():
        if isinstance(v, dict):
            logger.info(colorize(
                empty_space + ''.join([str(k).rjust(28), f" | {'*' * (depth + 1)} --->"]), color='blue'))
            show_dict(v, depth=depth + 1)
        else:
            logger.info(
                empty_space + ''.join([str(k).rjust(28), ' | ', str(v).ljust(28)]))

    if depth == 0:
        logger.info(colorize('-' * 80, color='blue', bold=True))

def save_config(dicpath: str, config: Dict, filename: str):
    if not os.path.exists(dicpath):
        os.makedirs(dicpath)
    with open(os.path.join(dicpath, filename), 'w', encoding='utf-8') as fw:
        yaml.dump(config, fw)
    logger.info(colorize(f'save config to {dicpath} successfully', color='green'))

def load_config(filename: str, not_find_error=True) -> Dict:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            x = yaml.safe_load(f.read())
        logger.info(
            colorize(f'load config from {filename} successfully', color='green'))
        return x or {}
    else:
        if not_find_error:
            raise Exception('cannot find this config.')
        else:
            logger.info(
                colorize(f'load config from {filename} failed, cannot find file.', color='red'))