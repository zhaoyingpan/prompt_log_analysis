import os
import tqdm
import math
import json
import csv
import argparse
import torch
import pandas as pd
import operator
import numpy as np
import itertools
from ast import literal_eval
from collections import Counter, OrderedDict

from utils import *

def term_stats(data_path):
    tokens = read_tokens_from_file(data_path)
    num_unique_terms = len(set(list(itertools.chain.from_iterable(tokens))))
    print('Number of unique terms: {}'.format(num_unique_terms))

    num_tokens_per_prompt = [len(t) for t in tokens]
    print('Statistics of number of tokens per prompt: {}'.format(num_tokens_per_prompt))
    _, _, _, _ = data_statistics(num_tokens_per_prompt)

def find_nondataset_word(dataset_name, data_path, laion_path, save_root):
    vocab = dict(Counter(list(itertools.chain.from_iterable(read_tokens_from_file(data_path)))))
    laion_vocab = dict(Counter(list(itertools.chain.from_iterable(read_tokens_from_file(laion_path)))))

    words = list(np.setdiff1d(list(vocab.keys()), list(laion_vocab.keys())))
    word_freq = OrderedDict(sorted({term: vocab[term] for term in words}.items(), key=lambda t: t[1], reverse=True))
    df = pd.DataFrame({'term': list(word_freq.keys()),
                       'freq': list(word_freq.values())})
    df.to_csv(os.path.join(save_root, '{}_oov_words.csv'.format(dataset_name)), index=False)
    return os.path.join(save_root, '{}_oov_words.csv'.format(dataset_name))

class Reweighting():
    def __init__(self, dataset_name, all_word_paths, all_pairs_path, reweighting_save_root, cut_off_k):
        self.dataset_name = dataset_name.lower()
        self.save_root = reweighting_save_root
        self.all_words_freqs = json.load(open(all_word_paths))
        all_term_pairs = json.load(open(all_pairs_path))

        self.dict_term_pairs = {literal_eval(pair[0]): pair[1] for pair in all_term_pairs}
        self.cut_off_pairs, self.vocab, self.normalized_matrix, self.PX = self.top_pairs_cut_off(cut_off_k)
    
    def top_pairs_cut_off(self, cut_off_k):
        term_rankings = list(self.all_words_freqs.keys())
        pair_rankings = list(self.dict_term_pairs.keys())

        cut_off_pairs = {}

        pair_ranking = -1
        vocab = []
        for pair, freq in self.dict_term_pairs.items():
            first_term = pair[0]
            second_term = pair[1]
            if first_term == ' ' or second_term == ' ':
                continue
            term_freq_1 = self.all_words_freqs[first_term]
            term_freq_2 = self.all_words_freqs[second_term]

            term1_ranking = term_rankings.index(first_term)
            term2_ranking = term_rankings.index(second_term)
            pair_ranking += 1

            if term1_ranking < cut_off_k and term2_ranking < cut_off_k and pair_ranking < cut_off_k:
                cut_off_pairs[pair] = freq
                vocab.append(first_term)
                vocab.append(second_term)
            if pair_ranking == cut_off_k - 1:
                break
        vocab = set(vocab)

        to_save_all_pairs = {}
        for key, value in cut_off_pairs.items():
            new_key = str(key)
            to_save_all_pairs[new_key] = value

        by_value_pairs = sorted(to_save_all_pairs.items(), key = lambda item:item[1], reverse=True)

        with open(os.path.join(self.save_root, '{}_cut_off_pairs.json'.format(self.dataset_name)), "w") as outfile:
            json.dump(by_value_pairs, outfile)

        matrix_size = len(vocab)
        vocab = list(vocab)
        count_matrix = torch.zeros(matrix_size,matrix_size)
        for i in tqdm.tqdm(range(count_matrix.shape[0])):
            for j in range(count_matrix.shape[1]):
                word1 = vocab[i]
                word2 = vocab[j]
            tup = tuple(sorted([word1, word2]))
            try:
                Pij = self.cut_off_pairs[tup]
            except:
                Pij = 0
            count_matrix[i,j] = Pij
        
        normalized_matrix = count_matrix/torch.sum(count_matrix)
        PX = torch.sum(normalized_matrix, dim=1, keepdim=False)
        
        return cut_off_pairs, vocab, normalized_matrix, PX

    def ours(self):
        for alpha in [0,1,3,5,10]:
            reweighted_pairs = {}
            for pair, freq in self.cut_off_pairs.items():
                first_term = pair[0]
                second_term = pair[1]
                term_freq_1 = self.all_words_freqs[first_term]
                term_freq_2 = self.all_words_freqs[second_term]
                reg = freq/((term_freq_1 * term_freq_2) ** (1/2))
                reweighted_freq = (reg ** alpha)*freq
                reweighted_pairs[pair] = (reweighted_freq, freq, term_freq_1, term_freq_2)
            reweighted_pairs_list = sorted(reweighted_pairs.items(), key=lambda t: (t[1][0],t[1][1]), reverse=True)
            csv_name = os.path.join(self.save_root, '{}_reweighted_pairs_ours_{}.csv'.format(self.dataset_name, str(alpha)))
            with open(csv_name, 'w') as f:
                write = csv.writer(f)
                for pair in reweighted_pairs_list:
                    write.writerow(pair)
       
        return csv_name
    
    def PMI(self):
        reweighted_pairs = {}
        for pair, freq in self.cut_off_pairs.items():
            first_term = pair[0]
            second_term = pair[1]
            term_freq_1 = self.all_words_freqs[first_term]
            term_freq_2 = self.all_words_freqs[second_term]
            reweighted_freq = math.log2(freq/(term_freq_1*term_freq_2))
            reweighted_pairs[pair] = (reweighted_freq, freq, term_freq_1, term_freq_2)
        
        reweighted_pairs_list = sorted(reweighted_pairs.items(), key=lambda t: (t[1][0],t[1][1]), reverse=True)
        csv_name = os.path.join(self.save_root, '{}_reweighted_pairs_PMI.csv'.format(self.dataset_name))
        with open(csv_name, 'w') as f:
            write = csv.writer(f)
            for pair in reweighted_pairs_list:
                write.writerow(pair)
        
        return csv_name
    
    def tstats(self):

        reweighted_pairs = {}

        for pair, freq in tqdm.tqdm(self.cut_off_pairs.items()):
            first_term = pair[0]
            second_term = pair[1]
            term1_idx = self.vocab.index(first_term)
            term2_idx = self.vocab.index(second_term)
            PXij = self.normalized_matrix[term1_idx, term2_idx]
            Reg = self.PX[term1_idx] * self.PX[term2_idx] 

            reweighted_freq = (PXij - Reg)/pow(Reg, 1/2)
        
            term_freq_1 = self.all_words_freqs[first_term]
            term_freq_2 = self.all_words_freqs[second_term]

            reweighted_pairs[pair] = (reweighted_freq, freq, term_freq_1, term_freq_2)
        
        reweighted_pairs_list = sorted(reweighted_pairs.items(), key=lambda t: (t[1][0],t[1][1]), reverse=True)
        csv_name = os.path.join(self.save_root, '{}_reweighted_pairs_ttest.csv'.format(self.dataset_name))
        with open(csv_name, 'w') as f:
            write = csv.writer(f)
            for pair in reweighted_pairs_list:
                write.writerow(pair)
        
        return csv_name
    
    def chi_square(self):
        reweighted_pairs = {}
        for pair, freq in tqdm.tqdm(self.cut_off_pairs.items()):
            first_term = pair[0]
            second_term = pair[1]
            try:#avoid tie 10k-th word
                term1_idx = self.vocab.index(first_term)
                term2_idx = self.vocab.index(second_term)
            except:
                continue
            O11 = self.normalized_matrix[term1_idx, term2_idx]
            O12 = self.PX[term1_idx] - O11
            O21 = self.PX[term2_idx] - O11
            O22 = 1 -  O11 - O12 - O21
            R1 = O11 + O12
            R2 = O21 + O22
            C1 = O11 + O21
            C2 = O12 + O22
            E11 = R1 * C1
            E12 = R1 * C2
            E21 = R2 * C1
            E22 = R2 * C2

            reweighted_freq = pow((O11 - E11),2)/E11 + pow((O12 - E12),2)/E12 + pow((O21 - E21),2)/E21 + pow((O22 - E22),2)/E22
        
            term_freq_1 = self.all_words_freqs[first_term]
            term_freq_2 = self.all_words_freqs[second_term]

            reweighted_pairs[pair] = (reweighted_freq, freq, term_freq_1, term_freq_2)
        
        reweighted_pairs_list = sorted(reweighted_pairs.items(), key=lambda t: (t[1][0],t[1][1]), reverse=True)
        csv_name = os.path.join(self.save_root, '{}_reweighted_pairs_chi-square.csv'.format(self.dataset_name))
        with open(csv_name, 'w') as f:
            write = csv.writer(f)
            for pair in reweighted_pairs_list:
                write.writerow(pair)
        
        return csv_name
    

def first_order(dataset_name, csv_path, save_root):
    data = pd.read_csv(csv_path)
    tokenized_list = data['tokenized'].tolist()

    all_terms = []
    for prompt in tqdm.tqdm(tokenized_list):
        words = literal_eval(prompt)
        words = list(set(words))
        all_terms.extend(words)

    all_words_freqs = dict(Counter(all_terms))
    all_words_freqs = dict(sorted(all_words_freqs.items(), key=operator.itemgetter(1), reverse=True))

    with open(os.path.join(save_root, "{}_all_words&freqs(with punctuation).json".format(dataset_name)), "w") as f:
        json.dump(all_words_freqs, f)

    # Sort the dictionary by values in descending order
    sorted_query_frequencies = all_words_freqs
    # sorted_query_frequencies = {word: math.log(value) for word, value in sorted_query_frequencies.items()}

    # Get the values for the y-axis
    y_values = list(sorted_query_frequencies.values())

    # Get the ranking for each query and use it as the x-values
    x_values = [i for i, _ in enumerate(y_values, start=1)]

    _, _, _, _ = data_statistics(all_words_freqs.values())

    with open(os.path.join(save_root, '{}_all_terms_by_freq.csv'.format(dataset_name)), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in all_words_freqs.items():
            writer.writerow([key, value])
    
    return os.path.join(save_root, "{}_all_words&freqs(with punctuation).json")

def second_order(dataset_name, csv_path, save_root):
    data = pd.read_csv(csv_path)
    tokenized_list = data['tokenized'].tolist()

    all_term_pairs = dict()
    for tokenized_query in tqdm(tokenized_list):
        tokenized_query = literal_eval(tokenized_query)
        words = list(set(tokenized_query))
        try:
            words.remove(' ')
        except:
            pass
        term_pairs = list(set([tuple(sorted([a, b])) for idx, a in enumerate(words) for b in words[idx + 1:]]))

        for pair in term_pairs:
            if pair not in all_term_pairs.keys():
                all_term_pairs[pair] = 0
            all_term_pairs[pair] += 1
    
    to_save_all_pairs = dict()
    for key, value in all_term_pairs.items():
        new_key = str(key)
        to_save_all_pairs[new_key] = value

    by_value_pairs = sorted(to_save_all_pairs.items(),key = lambda item:item[1], reverse=True)
    with open(os.path.join(save_root, "{}_all_pairs.json".format(dataset_name)), "w") as outfile:
        json.dump(by_value_pairs, outfile)

    return os.path.join(save_root, "{}_all_pairs.json".format(dataset_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--laion_path', required=True, type=str)
    parser.add_argument('--save_root', required=True, type=str)
    parser.add_argument('--reweighting_type', type=str, choices=['ours', 'PMI', 'tstats', 'chi-square'])
    parser.add_argument('--cut_off_k', type=int, default=10000)
    args = parser.parse_args()

    print('Basic statistics of {} (term-level):'.format(args.dataset_name))
    term_stats(args.data_path)

    print('Begin first/second-order analysis...')
    all_word_paths = first_order(args.dataset_name, args.csv_path, args.save_root)
    all_pairs_path = second_order(args.dataset_name, args.csv_path, args.save_root)
    reweight = Reweighting(args.dataset_name, all_word_paths, all_pairs_path, args.save_root, args.cut_off_k)
    reweight_method = reweight.get_method(args.reweighting_type)
    result_path = reweight_method()
    print('Saved frequency with words (first-order) results to {}. Saved reweighted results (second-order) to {}.'.format(all_word_paths, result_path))
    
    print('Begin to find out-of-vocabulary (oov) words')
    oov_path = find_nondataset_word(args.dataset_name, args.data_path, args.laion_path, args.save_root)
    print('Saved results to {}'.format(oov_path))