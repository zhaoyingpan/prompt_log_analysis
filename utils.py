import os
import statistics
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
from ast import literal_eval
from datetime import datetime

def power_law(x, a, b):
    return a * np.power(x, -b)

def get_unique_id(df):
    return list(set(df['author_id'].tolist()))

def merge_colons(words):
    new_words = []
    i = 0
    while(True):
        current_word = words[i]
        if current_word == ":":
            if i+1 < len(words):
                next_word = words[i+1]
                if current_word+next_word == "::":
                    current_word = current_word + next_word
                    i = i + 2
                else:
                    i = i + 1
            else:
                i = i + 1
        else:
            i = i + 1

        new_words.append(current_word)
        if i >= len(words):
            break
    return new_words

def merge_colons_minus(words):
    new_words = []
    i = 0
    while(True):
        current_word = words[i]
        if current_word == "::":
            if i+1 < len(words):
                next_word = words[i+1]
                if current_word+next_word == "::-":
                    current_word = current_word + next_word
                    i = i + 2
                else:
                    i = i + 1
            else:
                i = i + 1
        else:
            i = i + 1

        new_words.append(current_word)
        if i >= len(words):
            break
    return new_words

def tqdm_df(length, index):
    if index % (length // 10) == 0:
        print('completed {}%'.format(10 * (index // (length // 10))))


def tokenize(df, dataset_name):
    nlp = spacy.load('en_core_web_sm')
    cleaned_df = []
    tokenized_queries = []
    length = len(df)
    
    for index, row in df.iterrows():
        tqdm_df(length, index)
        prompt = row['prompt']
        if type(prompt) == str:
            doc = nlp(prompt)
            words = []
            for token in doc:
                if str(token) != ' ':
                    words.append(token.lower_)
            if dataset_name == 'midjourney':
                words = merge_colons(words)
                words = merge_colons_minus(words)
            if len(words) > 0:
                tokenized_queries.append(words)
                cleaned_df.append(row)
        else:
            continue
    cleaned_df = pd.DataFrame(cleaned_df)
    cleaned_df['tokenized'] = tokenized_queries
    return cleaned_df


def data_statistics(data, enable_print=True):
    if len(data) == 0:
        return [float('nan'), float('nan'), float('nan'), (float('nan'), float('nan'))]
        
    med = statistics.median(data)
    mean = statistics.mean(data)
    if len(data) > 1:
        std = statistics.stdev(data)
    else:
        std = float('nan')
    data_range = (min(data), max(data))
    if enable_print:
        print("median: {}, mean: {}, std: {}, range: {}".format(med, mean, std, data_range))
    return [med, mean, std, data_range]

def cut_freq_dict(freq_dict):
    new_dict = {}
    for key, value in freq_dict.items():
        if value > 1:
            new_dict[key] = value
    return new_dict

def cut_list_dict(freq_dict, value_index=None):
    new_dict = {}
    for key, value in freq_dict.items():
        if value_index is not None:
            if len(value) > 1:
                new_dict[key] = value
        else:
            if len(value[value_index]) > 1:
                new_dict[key] = value
    return new_dict

def remove_nan(data_list):
    new_list = [item for item in data_list if item == item]
    return new_list

def check_nan(value):
    if value == value:
        return True
    else:
        return False

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def create_root_folder(save_root):
    if os.path.exists(save_root):
        user_input = input('folder already exists, press ENTER to delete:')
        if user_input == '':
            shutil.rmtree(save_root)
        else:
            exit()
    os.mkdir(save_root)
    
def read_tokens(df):
    return [literal_eval(t) for t in df['tokenized'].tolist()]

def read_tokens_from_file(df_path):
    df = pd.read_csv(df_path)
    return [literal_eval(t) for t in df['tokenized'].tolist()]


def get_time_diff(timestamp1, timestamp2):

    fmt = '%Y-%m-%dT%H:%M:%S'
    tstamp1 = datetime.strptime(timestamp1, fmt)
    tstamp2 = datetime.strptime(timestamp2, fmt)
    td = tstamp2 - tstamp1
    td_mins = td.total_seconds() / 60
    return td_mins

def cut_timestamp_sec(timestamp, dataset_name):
    if dataset_name == 'midjourney':
        if '+' in timestamp:
            timestamp = timestamp.split('+')[0]
        if '.' in timestamp:
            timestamp = timestamp.split('.')[0]
    elif dataset_name == 'diffusiondb':
        timestamp = timestamp.split('+')[0].replace(' ', 'T')

    return timestamp
    
def cal_avg_time_intervals(df):
    timestamps = df['timestamp'].tolist()
    all_intervals = []
    for i, ts in enumerate(timestamps[1:]):
        all_intervals.append(get_time_diff(timestamps[i], ts))
    return np.array(all_intervals).mean()
