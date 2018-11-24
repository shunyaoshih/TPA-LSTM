import argparse
import json
import logging
import tensorflow as tf

from lib.utils import create_dir, check_path_exists


def params_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_set', type=str, default='muse')
    parser.add_argument('--decay', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--file_output', type=int, default=1)
    parser.add_argument('--highway', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--init_weight', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_gradient_norm', type=float, default=5.0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_dir', type=str, default='./models/model')
    parser.add_argument('--mts', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_units', type=int, default=338)

    para = parser.parse_args()

    if para.data_set == "muse" or para.data_set == "lpd5":
        para.mts = 0

    para.logging_level = logging.INFO

    if para.attention_len == -1:
        para.attention_len = para.max_len

    create_dir(para.model_dir)

    json_path = para.model_dir + '/parameters.json'
    json.dump(vars(para), open(json_path, 'w'), indent=4)
    return para


def logging_config_setup(para):
    if para.file_output == 0:
        logging.basicConfig(
            level=para.logging_level, format='%(levelname)-8s - %(message)s')
    else:
        logging.basicConfig(
            level=para.logging_level,
            format='%(levelname)-8s - %(message)s',
            filename=para.model_dir + '/progress.txt')
        logging.getLogger().addHandler(logging.StreamHandler())
    tf.logging.set_verbosity(tf.logging.ERROR)


def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config
