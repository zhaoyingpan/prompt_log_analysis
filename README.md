# A Prompt Log Analysis of Text-to-Image Generation Systems

Thanks for your interest in our WWW 2023 paper [A Prompt Log Analysis of Text-to-Image Generation Systems](https://dl.acm.org/doi/abs/10.1145/3543507.3587430). 
In this repository, we provide code for data preparation and prompt analysis. For the processed data that we used in the paper (Midjourney, DiffusionDB, SAC, and LAION), please visit the [Google Drive folder](https://drive.google.com/drive/folders/119Ib6XkaksCQdagmgoRLnqpzPFzRxjBJ?usp=sharing). For the analysis results and visualizations, please refer to the [Google Drive folder](https://drive.google.com/drive/folders/1iBYcQa2SoLFs6SF12BK3SgDTAZZVZv3x?usp=share_link) and see [Data File Descriptions](#data-files) for descriptions of the files.


## Table of Content

* [Datasets](#datasets)
* [Data Processing](#data-processing)
* [Prompt Analysis](#prompt-analysis)
  * [Term-level analysis](#term-level-analysis)
  * [Prompt-level analysis](#prompt-level-analysis)
  * [Session-level analysis](#session-level-analysis)
* [Data File Descriptions](#data-file-descriptions)
* [Citation and Contact](#citation-and-contact)

## Datasets

In our paper, we analyze user-input text-to-image prompts from three large-scale datasets:

* [Midjourney Discord Dataset](https://www.kaggle.com/datasets/da9b9ba35ffbd86a5f97ccd068d3c74f5742cfe5f34f6aaf1f0f458d7694f55e)
* [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)
* [Simulacra Aesthetic Captions (SAC)](https://github.com/JD-P/simulacra-aesthetic-captions)

Besides, we analyzed the [LAION-400M](https://www.kaggle.com/datasets/romainbeaumont/laion400m) text-image dataset, which is often been used to train text-to-image models. 

## Data Processing

Dataset processing includes extraction of basic fields (e.g., prompt, timestamp, and user ID) from raw datasets, prompt standardization, and tokenization. For details about dataset processing, please refer to the appendix of our paper.

To process the datasets we used in the paper, run the following command:

```python
python prepare_data.py --dataset_name <dataset_name> --raw_data_root <raw_dataset_root/path> --save_root <result_folder>
```

* `--dataset_name` supports inputting one dataset name (choosing from `midjourney`/`diffusiondb`/`sac`/`laion`)
* `--raw_data_root` is the root folder (for Midjourney and LAION) or file path (for DiffusionDB and SAC) of the downloaded raw datasets
* `--save_root` is the folder to save the processed data.

To process a subset of LAION-400M, the augment `--num_samples` needs to be specified as the number of samples. The default value is 1M:

```python
python prepare_data.py --dataset_name LAION --raw_data_root <raw_dataset_root/path> --save_root <result_folder> --num_samples 1000000
```

## Prompt Analysis

### Term-level Analysis

Term-level analysis contains basic statistics, first-order analysis, second-order analysis, and out-of-vocabulary (OOV) word analysis. For OOV analysis, please first conduct data processing for both the user-input prompt dataset (midjourney/diffusiondb/sac) and LAION.

```python
python term_level.py --dataset_name <dataset_name> --data_path <tokenized_file_path> --laion_path <tokenized_laion_file_path> --save_root <result_folder>
```

* `--reweighting_type` allows choosing different reweighting methods, including PMI, tstats and chi-square. We recommend using the default chi-square reweighting method, which also corresponds to the results presented in the paper.
* `--cut_off_k` adjusts the threshold of ranking for reweighting (10000 by default), selecting a limited number of terms and term pairs.

Note that it's recommended to create a new folder for `--save_root` to save the analysis result files.

### Prompt-level Analysis

Prompt-level analysis contains basic statistics, DiffusionDB prompt frequency ranking plotting, timestamp plotting, rating histogram plotting (only for SAC), and correlation calculation for ratings and prompt lengths (only for SAC). 

```python
python prompt_level.py --dataset_name <dataset_name> --data_path <tokenized_file_path> --save_root <result_folder>
```

* `--topk_prompts` adjusts the number of top frequent prompts to be saved in the file.

### Session-level Analysis

Session-level analysis contains session grouping, prompt popularity analysis, editing analysis and repeated prompt analysis. See more details about sessions in our paper. Note that since sessions are grouped with a threshold of time, session-level analysis only supports Midjourney and DiffusionDB.

```python
python session_level.py --dataset_name <dataset_name> --data_path <tokenized_file_path> --threshold <session_threshold> --save_root <result_folder>
```

* `--threshold` sets the time threshold for sessions, and the default value is 30 (min).


## Data File Descriptions

Our analysis results and visualizations are provided in the [Google Drive folder](https://drive.google.com/drive/folders/1iBYcQa2SoLFs6SF12BK3SgDTAZZVZv3x?usp=share_link). Here are the file descriptions.

| No. | Filename | Description |
| --- | --- | --- |
| 1 | Midjourney_tokenized.csv | Preprocessed and tokenized file for the Midjourney dataset. |
| 2 | DiffusionDB_tokenized.csv | Preprocessed and tokenized file for the DiffusionDB dataset. |
| 3 | SAC_tokenized.csv | Preprocessed and tokenized file for the SAC dataset. |
| 4 | LAION_tokenized.csv | Preprocessed and tokenized file for the LAION dataset (subset of 1M). |
| 5 | Midjourney_all_terms_by_freq.csv | List of all terms in the Midjourney dataset, ranked by frequency in descending order. Table 3 displays the top 20 terms. |
| 6 | DiffusionDB_all_terms_by_freq.csv | List of all terms in the DiffusionDB dataset, ranked by frequency in descending order. Table 3 displays the top 20 terms. |
| 7 | SAC_all_terms_by_freq.csv | List of all terms in the SAC dataset, ranked by frequency in descending order. Table 3 displays the top 20 terms. |
| 8 | Midjourney_reweighted_pairs.csv | List of all term pairs in the Midjourney dataset, ranked by frequency in descending order. Table 4 displays the top 20 pairs. |
| 9 | DiffusionDB_reweighted_pairs.csv | List of all term pairs in the DiffusionDB dataset, ranked by frequency in descending order. Table 4 displays the top 20 pairs. |
| 10 | SAC_reweighted_pairs.csv | List of all term pairs in the SAC dataset, ranked by frequency in descending order. Table 4 displays the top 20 pairs. |
| 11 | DiffusionDB_prompt_freq.csv | Frequencies of prompts in the DiffusionDB dataset (only includes entries with a frequency greater than 1). Table 5 displays the top 20 most frequent prompts. |
| 12 | DiffusionDB_prompt_userfreq.csv | Prompts shared across users in the DiffusionDB dataset (only includes entries with a frequency greater than 1). Some of the most shared prompts are listed in Table 7. |
| 13 | Midjourney_replaced_freq.csv | Frequencies of term replacements in sessions of the Midjourney dataset. Table 9 displays the top 30 most frequent replacements. |
| 14 | DiffusionDB_replaced_freq.csv | Frequencies of term replacements in sessions of the DiffusionDB dataset. Table 9 displays the top 30 most frequent replacements. |
| 15 | SAC_term_avg_rating.csv | Terms ranked by average ratings in descending order. |
| 16 | Midjourney_oov_terms.csv | Out-of-vocabulary (OOV) terms in the Midjourney dataset. |
| 17 | DiffusionDB_oov_terms.csv | Out-of-vocabulary (OOV) terms in the DiffusionDB dataset. |
| 18 | SAC_oov_terms.csv | Out-of-vocabulary (OOV) terms in the SAC dataset. |
| 19 | Midjourney_word_embedding_top5000.html | HTML file displaying the visualization of term embeddings for the most frequent 5000 terms in the Midjourney dataset. The embeddings are obtained using BERT and reduced to 2 dimensions using UMAP. Hover over data points to view corresponding terms. |
| 20 | DiffusionDB_word_embedding_top5000.html | HTML file displaying the visualization of term embeddings for the most frequent 5000 terms in the DiffusionDB dataset. The embeddings are obtained using BERT and reduced to 2 dimensions using UMAP. Hover over data points to view corresponding terms. |
| 21 | SAC_word_embedding_top5000.html | HTML file displaying the visualization of term embeddings for the most frequent 5000 terms in the SAC dataset. The embeddings are obtained using BERT and reduced to 2 dimensions using UMAP. Hover over data points to view corresponding terms. |

In our [data files](https://drive.google.com/drive/folders/119Ib6XkaksCQdagmgoRLnqpzPFzRxjBJ?usp=sharing), all files follow a similar structure with common columns, such as `prompt` (cleaned prompts) and `tokenized` (tokenized prompts). Additionally, each dataset-specific file includes certain columns for analysis purposes, while other information is provided for reference. Here is a breakdown of the columns in each file:

- **Midjourney_tokenized.csv**:
  - `prompt`: Cleaned prompts.
  - `tokenized`: Tokenized prompts.
  - `url`: Links to generated images (may expire due to Discord's policy).
  - `timestamp`: Timestamp of the prompt.
  - `author_id`: ID of the prompt author.
  - `username`: Username of the prompt author.

- **DiffusionDB_tokenized.csv**:
  - `prompt`: Cleaned prompts.
  - `tokenized`: Tokenized prompts.
  - `author_id`: ID of the prompt author.
  - `timestamp`: Timestamp of the prompt.
  - `image_name`: Name of the associated image.

- **SAC_tokenized.csv**:
  - `prompt`: Cleaned prompts.
  - `tokenized`: Tokenized prompts.
  - `pid`: Prompt ID. One prompt can have multiple `pid` entries.
  - `method`: Type of the method used to generate images corresponding to the prompt.
  - `iid`: Image ID associated with the prompt.
  - `path`: Path where the image is saved.
  - `rating`: Rating given to the prompt.

- **LAION_tokenized.csv**:
  - `prompt`: Cleaned prompts.
  - `tokenized`: Tokenized prompts.
  - `sample_id`: ID of the prompt sample.
  - `image_url`: URL of the associated image.

Please note that for the analysis, only the `timestamp`, `author_id`, and `rating` columns are utilized, while the other columns provide additional information for reference.

## Contact and Citation

Please feel free to contact us if you have any questions.

```
@inproceedings{10.1145/3543507.3587430,
    author = {Xie, Yutong and Pan, Zhaoying and Ma, Jinge and Jie, Luo and Mei, Qiaozhu},
    title = {A Prompt Log Analysis of Text-to-Image Generation Systems},
    year = {2023},
    isbn = {9781450394161},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3543507.3587430},
    doi = {10.1145/3543507.3587430},
    abstract = {Recent developments in large language models (LLM) and generative AI have unleashed the astonishing capabilities of text-to-image generation systems to synthesize high-quality images that are faithful to a given reference text, known as a “prompt”. These systems have immediately received lots of attention from researchers, creators, and common users. Despite the plenty of efforts to improve the generative models, there is limited work on understanding the information needs of the users of these systems at scale. We conduct the first comprehensive analysis of large-scale prompt logs collected from multiple text-to-image generation systems. Our work is analogous to analyzing the query logs of Web search engines, a line of work that has made critical contributions to the glory of the Web search industry and research. Compared with Web search queries, text-to-image prompts are significantly longer, often organized into special structures that consist of the subject, form, and intent of the generation tasks and present unique categories of information needs. Users make more edits within creation sessions, which present remarkable exploratory patterns. There is also a considerable gap between the user-input prompts and the captions of the images included in the open training data of the generative models. Our findings provide concrete implications on how to improve text-to-image generation systems for creation purposes.},
    booktitle = {Proceedings of the ACM Web Conference 2023},
    pages = {3892–3902},
    numpages = {11},
    keywords = {Query Log Analysis., Prompt Analysis, AI-Generated Content (AIGC), AI for Creativity, Text-to-Image Generation},
    location = {Austin, TX, USA},
    series = {WWW '23}
}
```
