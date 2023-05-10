import os
import tqdm
import pandas as pd
import numpy as np
import argparse
import editdistance
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

from utils import *

def df_mark_session(dataset_name, data_path, threshold, save_root):

    df = pd.read_csv(data_path)
    df = df.sort_values(['author_id', 'timestamp'])
    user_zfill = len(str(len(set(df['author_id'].tolist()))))
    length = len(df)
    user_list = []
    session_list = []
    num_queries = []
    lengths_per_user = []
    time_interval = []

    for i, (_, row) in enumerate(df.iterrows()):
        tqdm_df(length, i)
        author_id = row['author_id']
        ts = cut_timestamp_sec(row['timestamp'], dataset_name)
        if i == 0:
            prev_id = author_id
            prev_ts = ts
            user_idx = 0
            session_idx = 0
            session_list.append(str(user_idx).zfill(user_zfill)+'_'+str(session_idx))
            user_list.append(str(user_idx).zfill(user_zfill))
            interval = []
            num_q = 1
        else:
            if author_id == prev_id:
                time_diff = get_time_diff(prev_ts, ts)
                if time_diff > threshold:
                    prev_ts = ts
                    session_idx += 1
                    num_queries.append(num_q)
                    if len(interval) > 0:
                        time_interval.append((num_q, np.array(interval).mean()))
                    num_q = 1
                    interval = []
                else:
                    num_q += 1
                    interval.append(time_diff)

            else:
                prev_id = author_id
                prev_ts = ts
                lengths_per_user.append(session_idx+1)
                num_queries.append(num_q)
                if len(interval) > 0:
                    time_interval.append((num_q, np.array(interval).mean()))
                user_idx += 1
                num_q = 1
                interval = []
                session_idx = 0

            session_list.append(str(user_idx).zfill(user_zfill)+'_'+str(session_idx))
            user_list.append(str(user_idx).zfill(user_zfill))
            prev_ts = ts
        
    lengths_per_user.append(session_idx+1)
    num_queries.append(num_q)
    if len(interval) > 0:
        time_interval.append((num_q, np.array(interval).mean()))
    df['session'] = session_list
    df['user'] = user_list

    print('statistics of number of queries per session')
    _, _, _, _ = data_statistics(num_queries)
    print('statistics of number of sessions per user')
    _, _, _, _ = data_statistics(lengths_per_user)

    fn = os.path.join(save_root, '{}_session_{}min.csv'.format(dataset_name, threshold))
    df.to_csv(fn, index=False)

    return fn

def get_edit_distance(session_df_path):
    df = pd.read_csv(session_df_path)
    all_edit_distance = []
    for session_idx in tqdm.tqdm(set(df['session'].tolist())):
        data = df.loc[df['session'] == session_idx]
        tokens = read_tokens(data)
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                all_edit_distance.append(editdistance.eval(tokens[i], tokens[i+1]))
        else:
            continue
    _, _, _, _ = data_statistics(all_edit_distance)

def find_all_repeat(dataset_name, session_df_path, save_root):
    session_df = pd.read_csv(session_df_path)
    threshold = os.path.basename(session_df_path).split('min')[-2].split('_')[-1]
    freq_dict = {}
    all_sessions = list(set(session_df['session'].tolist()))
    for session_idx in tqdm.tqdm(all_sessions):
        data = session_df.loc[session_df['session'] == session_idx]
        tokens = read_tokens(data)
        for t in tokens:
            p = ' '.join(t)
            if p not in freq_dict.keys():
                freq_dict[p] = [1, [session_idx]]
            else:
                freq_dict[p][0] += 1
                freq_dict[p][1].append(session_idx)
    freq_dict = cut_list_dict(dict(OrderedDict(sorted(freq_dict.items(), key=lambda t: t[1][0], reverse=True))), value_index=1)
    
    num_users = []
    num_per_user = []
    stats = []
    for value in freq_dict.values():
        temp = dict(OrderedDict(sorted(dict(Counter([s_id.split('_')[0] for s_id in value[1]])).items(), key=lambda t: t[0], reverse=True)))
        num_users.append(len(temp))
        num_per_user.append(list(temp.values()))
        stats.append(data_statistics(list(temp.values()), enable_print=False))

    df = pd.DataFrame({"prompt": list(freq_dict.keys()),
                    "overall_freq": [value[0] for value in list(freq_dict.values())],
                    "num_users": num_users,
                    "num_per_user": num_per_user,
                    "stats": stats,
                    "session_idx": [list(set(value[1])) for value in list(freq_dict.values())],
                    "all_session_idx": [value[1] for value in list(freq_dict.values())]
                        })
    
    fn = os.path.join(save_root, '{}_overall_repeats_{}min.csv'.format(dataset_name, threshold))
    df.to_csv(fn, index=False)
    
    return fn

def freq_across_user(dataset_name, repeats_df_path, save_root):
    repeats_df = pd.read_csv(repeats_df_path)
    threshold = os.path.basename(repeats_df_path).split('min')[-2].split('_')[-1]
    df = repeats_df[['prompt', 'num_users']]
    df.sort_values('num_users', ascending=False)

    num_users = df['num_users'].tolist()
    plt.figure(figsize=(15,6))
    _ = plt.hist(num_users, bins=50, width=5)
    plt.yscale('log')
    plt.xticks(np.linspace(0, 400, 9))
    plt.grid(alpha=0.5)
    plt.xlabel('Shared by #users')
    plt.ylabel('#Prompts (log-scale)')

    fn = os.path.join(save_root, '{}_top100_prompt_across_users_{}min.csv'.format(dataset_name, threshold))
    df.head(100).to_csv(fn, index=False)
    return fn

def find_repeat_word(tokens1, tokens2):
    tokens1_freq = dict(Counter(tokens1))
    tokens2_freq = dict(Counter(tokens2))

    if abs(len(tokens1) - len(tokens2)) == 1:
        diff = set(tokens1_freq.items()) ^ set(tokens2_freq.items())
        return list(diff)[0][0]
    else:
        edit1 = None
        edit2 = None
        for t, freq1 in tokens1_freq.items():
            if t in tokens2_freq.keys():
                freq2 = tokens2_freq[t]
                if freq1 - freq2 == 1:
                    edit1 = t
                if freq2 - freq1 == 1:
                    edit2 = t
            else:
                edit1 = t
        if edit1 is not None and edit2 is None:
            for t, freq2 in tokens2_freq.items():
                if t not in tokens1_freq.keys():
                    edit2 = t
        return [edit1, edit2]

def rank_edits(edit_df, edit_name):
    freq = dict(OrderedDict(sorted(dict(Counter(edit_df[edit_name].tolist())).items(), key=lambda t: t[1], reverse=True)))
    df = pd.DataFrame({edit_name: list(freq.keys()),
                       'freq': list(freq.values())})
    return df

def rank_replace(replaced_df):
    freq = {}
    for item in replaced_df['replaced'].tolist():
        if tuple(item) not in freq.keys():
            freq[tuple(item)] = 1
        else:
            freq[tuple(item)] += 1
    freq = dict(OrderedDict(sorted(freq.items(), key=lambda t: t[1], reverse=True)))
    df = pd.DataFrame({'replaced': list(freq.keys()),
                       'freq': list(freq.values())})
    return df

def find_editted_word(dataset_name, data_path, threshold, save_root):
    df = pd.read_csv(data_path)
    df = df.sort_values(['author_id', 'timestamp'])
    user_zfill = len(str(len(set(df['author_id'].tolist()))))
    length = len(df)
    session_list = []
    added = []
    deleted = []
    replaced = []
    all_edits = []
    edit_type = []

    for i, (_, row) in enumerate(df.iterrows()):
        # tqdm_df(length, i)
        author_id = row['author_id']
        ts = cut_timestamp_sec(row['timestamp'], dataset_name)
        tokens = literal_eval(row['tokenized'])

        if i == 0:
            prev_id = author_id
            prev_ts = ts
            prev_tokens = tokens
            user_idx = 0
            session_idx = 0
            session_list.append(str(user_idx).zfill(user_zfill)+'_'+str(session_idx))
            all_edits.append('nan')
            edit_type.append('nan')
        else:
            if author_id == prev_id:
                time_diff = get_time_diff(prev_ts, ts)
                if time_diff > threshold:
                    session_idx += 1
                    all_edits.append('nan')
                    edit_type.append('nan')
                else:
                    if editdistance.eval(prev_tokens, tokens) == 1:
                        edits = find_repeat_word(prev_tokens, tokens)
                        if len(tokens) - len(prev_tokens) == 1:
                            added.append((' '.join(prev_tokens), ' '.join(tokens), edits))
                            all_edits.append(edits)
                            edit_type.append('add')
                        elif len(prev_tokens) - len(tokens) == 1:
                            deleted.append((' '.join(prev_tokens), ' '.join(tokens), edits))
                            all_edits.append(edits)
                            edit_type.append('delete')
                        else:
                            if not 'https://s.mj.run/' in edits[0] and not 'https://s.mj.run/' in edits[1]:
                                replaced.append((' '.join(prev_tokens), ' '.join(tokens), edits))
                                all_edits.append(edits)
                                edit_type.append('replace')
                            else:
                                all_edits.append('nan')
                                edit_type.append('nan')

                    else:
                        all_edits.append('nan')
                        edit_type.append('nan')
            else:
                prev_id = author_id
                user_idx += 1
                session_idx = 0
                all_edits.append('nan')
                edit_type.append('nan')

            session_list.append(str(user_idx).zfill(user_zfill)+'_'+str(session_idx))
            prev_ts = ts
            prev_tokens = tokens
    
    add_df = pd.DataFrame({'prompt1': [list(item)[0] for item in added],
                       'prompt2': [list(item)[1] for item in added],
                       'added': [list(item)[2] for item in added]})
    deleted_df = pd.DataFrame({'prompt1': [list(item)[0] for item in deleted],
                       'prompt2': [list(item)[1] for item in deleted],
                       'deleted': [list(item)[2] for item in deleted]})
    replaced_df = pd.DataFrame({'prompt1': [list(item)[0] for item in replaced],
                       'prompt2': [list(item)[1] for item in replaced],
                       'replaced': [list(item)[2] for item in replaced]})
    
    rank_edits(add_df, 'added').to_csv(os.path.join(save_root, '{}_added_freq.csv'.format(dataset_name)), index=False)
    rank_edits(deleted_df, 'deleted').to_csv(os.path.join(save_root, '{}_deleted_freq.csv'.format(dataset_name)), index=False)
    fn = os.path.join(save_root, '{}_replaced_freq.csv'.format(dataset_name))
    rank_replace(replaced_df).to_csv(fn, index=False)
    df['edits'] = all_edits
    df['edits_type'] = edit_type
    df.to_csv(os.path.join(save_root, '{}_edits.csv'.format(dataset_name)))
    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, type=str, choices=['midjourney', 'diffusiondb'])
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--save_root', required=True, type=str)
    parser.add_argument('--threshold', required=True, type=int, default=30)
    args = parser.parse_args()

    print('Begin to group by sessions with threhold of {}min...'.format(args.threshold))
    session_df_path = df_mark_session(args.dataset_name, args.data_path, args.threshold, args.save_root)
    print('Results saved to {}.'.format(session_df_path))

    print('Begin to find repeats...'.format(args.threshold))
    repeats_df_path = find_all_repeat(args.dataset_name, session_df_path, args.save_root)
    print('Results saved to {}.'.format(repeats_df_path))

    print('Begin to rank most frequent prompts across user...')
    top_prompts_across_user_path = freq_across_user(args.dataset_name, repeats_df_path, args.save_root)
    print('Results saved to {}.'.format(top_prompts_across_user_path))

    print('Statistics of edit distance within sessions...')
    get_edit_distance(session_df_path)

    print('Begin to find editted word (added/deleted/replaced) within sessions...')
    replacement_path = find_editted_word(args.dataset_name, args.data_path, args.threshold, args.save_root)
    print('Results of word replacements saved to {}.'.format(replacement_path))
