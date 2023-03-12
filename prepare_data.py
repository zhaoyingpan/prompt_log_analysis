import os
import argparse
import pandas as pd
from urllib.request import urlretrieve

from preprocess.midjourney_prep import *
from preprocess.sac_prep import *
from preprocess.diffusiondb_prep import *
from preprocess.laion_prep import *

from utils import *

def get_raw_csv(dataset_name, raw_data_root, save_root, num_samples=None):
    if dataset_name.lower() == 'midjourney':
        raw_csv_path = get_midjourney(raw_data_root, save_root)
    elif dataset_name.lower() == 'sac':
        raw_csv_path = get_sac(raw_data_root, save_root)
    elif dataset_name.lower() == 'diffusiondb':
        raw_csv_path = get_diffusiondb(raw_data_root, save_root)
    elif dataset_name.lower() == 'laion':
        raw_csv_path = get_laion(num_samples, raw_data_root, save_root)
    else:
        raise Exception('Please check the input dataset name! Choices: Midjourney, SAC, DiffusionDB, LAION. (case insensitive)')
    return raw_csv_path

def get_tokenized(dataset_name, raw_csv_path, save_root):
    if dataset_name.lower() in ['midjourney', 'sac', 'diffusiondb', 'laion']:
        data_file = pd.read_csv(raw_csv_path)[:100]
        tokenized_file = tokenize(data_file, dataset_name)
        fn = os.path.join(save_root, '{}_tokenized.csv'.format(dataset_name))
        tokenized_file.to_csv(fn, index=False)
        return fn

    else:
        raise Exception('Please check the input dataset name! Choices: Midjourney, SAC, DiffusionDB, LAION. (case insensitive)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--raw_data_root', required=True, type=str)
    parser.add_argument('--save_root', required=True, type=str)
    parser.add_argument('--num_samples', type=int, default=1000000)
    args = parser.parse_args()

    print('Begin to preprocess dataset: {}'.format(args.dataset_name))
    raw_csv_path = get_raw_csv(args.dataset_name, args.raw_data_root, args.save_root, args.num_samples)
    print('Raw csv file saved to {}!'.format(raw_csv_path))

    print('Begin tokenization of dataset: {}'.format(args.dataset_name))
    tokenized_path = get_tokenized(args.dataset_name, raw_csv_path, args.save_root)
    print('Tokenized file saved to {}!'.format(tokenized_path))
