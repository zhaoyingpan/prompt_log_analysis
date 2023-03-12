import os
import pandas as pd
import numpy as np
import argparse
import itertools
import matplotlib.dates as dates
from ast import literal_eval
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

from utils import *

def prompt_stats(dataset_name, data_path):
    df = pd.read_csv(data_path)
    print('Number of prompts: {}. Number of unique prompts: {}'.format(len(df), len(set(df['prompt'].tolist()))))
    if dataset_name.lower() in ['midjourney', 'diffusiondb']:
        print('Number of users: {}'.format(len(set(df['author_id'].tolist()))))

def plot_prompt_freq(dataset_name, session_df_path, save_root, num=100):
    session_df = pd.read_csv(session_df_path)
    df = session_df[['prompt', 'overall_freq']]
    df.sort_values('overall_freq', ascending=False)

    freq = df['overall_freq'].tolist()
    plt.figure(figsize=(10,5))
    plt.loglog(np.arange(len(freq))+1, freq, label='DiffusionDB', linewidth=3)
    popt, pcov = curve_fit(power_law, np.arange(len(freq))+1, freq, p0=[1, 1], bounds=[[1e-3, 1e-3], [1e20, 50]])
    plt.plot(np.arange(len(freq))+1, power_law(np.arange(len(freq))+1, *popt), label='Power law fit of DiffusionDB',  linestyle = "dashed",color="tab:blue", alpha=0.5, linewidth=3)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel('Prompt ranked by frequency (log-scale)')
    plt.ylabel('Prompt frequency (log-scale)')

    df.head(num).to_csv(os.path.join(save_root, '{}_top_prompts.csv'.format(dataset_name)), index=False)
    return os.path.join(save_root, '{}_top_prompts.csv'.format(dataset_name))

def plot_timestamp_hist_24h(dataset_name, data_path):
    df = pd.read_csv(data_path)
    timestamp_list = df['timestamp'].tolist()
    
    fig, ax = plt.subplots(figsize=(15,6))
    
    if dataset_name.lower() == 'midjourney':
        all_times = ['2000-01-01T'+t.split(':')[0].split('T')[1] for t in timestamp_list]
    elif dataset_name.lower() == 'diffusiondb':
        all_times = ['2000-01-01T'+t.split(':')[0].split(' ')[1] for t in timestamp_list]
    times_freq = dict(Counter(all_times))
    times_freq = OrderedDict(sorted(times_freq.items(), key=lambda t: t[0]))

    freq_density = np.array(list(times_freq.values())) / np.array(list(times_freq.values())).sum()
    timestamps = np.array(list(times_freq.keys()), dtype='datetime64')
    plt.plot(timestamps, freq_density, label=dataset_name.lower())

    ax.xaxis.set_major_locator(dates.HourLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    xticks = plt.gca().xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    xticks[-1].set_visible(False)

    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Proportion of prompts')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.legend()
    plt.grid(alpha=0.5)

def plot_rating_hist(sac_path):
    sac_file = pd.read_csv(sac_path)
    merged_ratings = [list(itertools.chain.from_iterable(literal_eval(r))) for r in sac_file['rating'].tolist()]
    all_ratings = list(itertools.chain.from_iterable(merged_ratings))

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(1, 1, 1)
    ax.margins(x=0)
    ax.hist(all_ratings, bins=np.arange(1, 12)-0.3, width=0.6, density=True, color='tab:blue')
    major_ticks = [0, 0.05, 0.1, 0.15]
    minor_ticks = 0.025 * np.arange(7)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(alpha=0.5, axis='y', which='both')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Density")
    ax.set_xlim((0.5, 10.5))
    ax.set_xticks(np.arange(1, 10.5))

def plot_rating_vs_length(sac_path):
    sac_file = pd.read_csv(sac_path)
    lengths = []
    ratings = []
    prompts = []
    for index, row in sac_file.iterrows():
        rs = list(itertools.chain.from_iterable(literal_eval(row['rating'])))
        if len(rs) > 0:
            for r in rs:
                prompts.append(row['prompt'])
                lengths.append(len(literal_eval(row['tokenized'])))
                ratings.append(r)
    plt.figure(figsize=(10, 6))

    s = sns.regplot(lengths, ratings, x_bins=50, color='tab:blue')
    s.set_xticklabels(s.get_xticks())
    s.set_yticklabels(s.get_yticks())
    plt.grid(alpha=0.5)
    plt.xlabel('Prompt length')
    plt.ylabel('Rating')

    pearson_corr, _ = stats.pearsonr(lengths, ratings)
    return pearson_corr
    # print('Pearson correlation coefficient: {}'.format(pearson_corr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--session_df_path', required=True, type=str)
    parser.add_argument('--save_root', required=True, type=str)
    parser.add_argument('--threshold', required=True, type=int, default=30)
    parser.add_argument('--topk_prompts', type=int, default=100)
    args = parser.parse_args()

    print('Basic statistics of {} (prompt-level):'.format(args.dataset_name))
    prompt_stats(args.dataset_name, args.data_path)

    print('Plot prompt frequency:')
    prompt_freq_path = plot_prompt_freq(args.dataset_name, args.session_df_path, args.save_root, args.topk_prompts)
    print('Results of prompts with frequency saved to {}.'.format(prompt_freq_path))

    print('Plot histogram of timestamps (per 24h):')
    plot_timestamp_hist_24h(args.dataset_name, args.data_path)
    
    if args.dataset_name.lower == 'sac':
        print('Plot histogram of ratings (SAC):')
        plot_rating_hist(args.data_path)

        print('Plot of ratings and prompt lengths (SAC):')
        pearson_corr = plot_rating_vs_length(args.data_path)
        print('The Pearson correlation coefficient is {}.'.format(pearson_corr))
