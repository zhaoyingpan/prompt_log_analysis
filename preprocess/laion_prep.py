import os
import tqdm
import pandas as pd
import sys
sys.path.append('..')
from utils import *

# laion-specific functions

def get_laion(num_sample, file_root, save_root):
    sampled_df_list = []
    num_files = 32
    for i in tqdm.tqdm(range(num_files)):
        filename = 'part-{}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'.format(str(i).zfill(5))
        data = pd.read_parquet(os.path.join(file_root, filename))
        data = data.sample(n=int(num_sample/num_files))
        df = pd.DataFrame({"prompt": data['TEXT'].tolist(),
                           "sample_id": data['SAMPLE_ID'].tolist(),
                           "image_url": data['URL'].tolist()
                        })
        sampled_df_list.append(df)
        del data, df

    data_file = pd.concat(sampled_df_list, ignore_index=True)
    data_file.to_csv(os.path.join(save_root, 'laion_raw.csv'), index=False)
    return os.path.join(save_root, 'laion_raw.csv')