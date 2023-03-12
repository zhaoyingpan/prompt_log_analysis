import os
import sys
sys.path.append('..')
import pandas as pd
from utils import *
# diffusiondb-specific functions

def get_diffusiondb(raw_data_root, save_root):
    data = pd.read_parquet(raw_data_root)
    cleaned_data = {}
    length = len(data)
    for index, row in data.iterrows():
        tqdm_df(length, index)
        prompt = row['prompt']
        author_id = row['user_name']
        timestamp = row['timestamp']
        cleaned_data[(prompt, author_id, timestamp)] = row['image_name']
    
    data_file = pd.DataFrame({"prompt": [key[0] for key, _ in cleaned_data.items()],
                              "author_id": [key[1] for key, _ in cleaned_data.items()],
                              "timestamp": [key[2] for key, _ in cleaned_data.items()],
                              "image_name": list(cleaned_data.values())
                                    })

    data_file.to_csv(os.path.join(save_root, 'diffusiondb_raw.csv'), index=False)
    return os.path.join(save_root, 'diffusiondb_raw.csv')