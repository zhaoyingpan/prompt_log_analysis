import pandas as pd
import sqlite3
import sys
sys.path.append('..')
from utils import *
# sac-specific functions

def get_sac(sqlite_path, save_path):
    # Create a SQL connection to the SQLite database
    con = sqlite3.connect(sqlite_path)

    cur = con.cursor()
    cur.execute("SELECT NAME FROM sqlite_master WHERE type='table';")

    tables = cur.fetchall()

    survey = pd.read_sql("SELECT * FROM survey", con=con)
    generations = pd.read_sql("SELECT * FROM generations", con=con)
    images = pd.read_sql("SELECT * FROM images", con=con)
    sqlite_sequence = pd.read_sql("SELECT * FROM sqlite_sequence", con=con)
    paths = pd.read_sql("SELECT * FROM paths", con=con)
    ratings = pd.read_sql("SELECT * FROM ratings", con=con)
    upscales = pd.read_sql("SELECT * FROM upscales", con=con)

    # match prompts with images and ratings
    prompts = generations["prompt"].tolist()
    prompt_id = generations["id"].tolist()
    prompt_dict = dict(zip(prompt_id, prompts))
    
    prompt_id = [int(p.split('_')[0]) for p in paths['path'].tolist()]
    df = paths.copy()
    df['pid'] = prompt_id

    raw_data = {}
    for pid, prompt in prompt_dict.items():
        method = generations.loc[generations['id'] == pid]['method'].tolist()
        iid_list = df.loc[df['pid'] == pid]['iid'].tolist()
        path_list = df.loc[df['pid'] == pid]['path'].tolist()
        rating_list = ratings.loc[ratings['iid'].isin(iid_list)]['rating'].tolist()
        if len(prompt) > 0:
            prompt = prompt.lower()
            if prompt not in raw_data.keys():
                raw_data[prompt] = {'pid': [pid],
                                    'method': [method],
                                    'iid' : [iid_list],
                                    'path': [path_list],
                                    'rating': [rating_list]
                                    }
            else:
                raw_data[prompt]['pid'].append(pid)
                raw_data[prompt]['method'].append(method)
                raw_data[prompt]['iid'].append(iid_list)
                raw_data[prompt]['path'].append(path_list)
                raw_data[prompt]['rating'].append(rating_list)
    
    data_file = pd.DataFrame({"prompt": list(raw_data.keys()),
                    "pid": [value['pid'] for value in raw_data.values()],
                    "method": [value['method'] for value in raw_data.values()],
                    "iid": [value['iid'] for value in raw_data.values()],
                    "path": [value['path'] for value in raw_data.values()],
                    "rating": [value['rating'] for value in raw_data.values()],
                    })
    data_file.to_csv(save_path, index=False)