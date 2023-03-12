# A Prompt Log Analysis of Text-to-Image Generation Systems

## About
We provide code and data files for our paper *A Prompt Log Analysis of Text-to-Image Generation Systems*, including dataset preparation, term-level analysis, prompt-level analysis, and session-level analysis. See data files and visualizations at [Google drive](https://drive.google.com/drive/folders/1iBYcQa2SoLFs6SF12BK3SgDTAZZVZv3x?usp=share_link)

## Table of Content
* [Dataset](#dataset)
* [Dataset preparation](#dataset-preparation)
* [Term-level analysis](#term-level)
* [Prompt-level analysis](#prompt-level)
* [Session-level analysis](#session-level)
* [Cite](#cite)
* [Contact](#contact)

## Dataset
Our code supports the following three dataset of user-input prompts for large text-to-image models. You are also welcome to follow the method to process or analyze future dataset in the same way.

* [Midjourney Discord dataset](https://www.kaggle.com/datasets/da9b9ba35ffbd86a5f97ccd068d3c74f5742cfe5f34f6aaf1f0f458d7694f55e)

* [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)

* [Simulacra Aesthetic Captions (SAC)](https://github.com/JD-P/simulacra-aesthetic-captions)

Besides, we also support the LAION-400M dataset, the dataset for training text-to-image models.

* [LAION-400M](https://www.kaggle.com/datasets/romainbeaumont/laion400m)

## Dataset preparation
Dataset preparation includes extraction of useful information (prompt, timestamp, user ID, etc.) from raw dataset, data cleaning for prompts, and tokenization for prompts. More details about dataset preprocessing can be found in the appendix of our paper.

To conduct dataset preparation, please git clone our repo, download the dataset according to the instruction of the dataset, and run this command

```python
python prepare_data.py --dataset_name <dataset_name> --raw_data_root <root of raw dataset> --save_root <path of processed data files>
```
where `--dataset_name` supports inputting one dataset name, choosing from `Midjourney/DiffusionDB/SAC/LAION` (case insensitive), `--raw_data_root` is the root folder of the downloaded raw dataset, and the `--save_root` is the folder where you wish to save the processed data.

To prepare the subset from LAION-400M, please specify the augment `--num_samples` to set the number of samples in the subset. The default value is 1M. The example command

```python
python prepare_data.py --dataset_name LAION --raw_data_root <root of raw dataset> --save_root <path of processed data files> --num_samples 1000000
```

## Term-level analysis
Term-level analysis contains basic statistics, first-order analysis, second-order analysis, and out-of-vocabulary (OOV) word analysis. For OOV analysis, please first conduct data preparation for both the user-input prompt dataset (midjourney/diffusiondb/sac) and subset from LAION.

```python
python term_level.py --dataset_name <dataset_name> --data_path <the path of the tokenized file> --laion_path <the path of the tokenized file of LAION subset> --save_root <path of result data files>
```
It's recommended to create a new folder for `--save_root` to save the analysis result files.

`--reweighting_type` allows to choose different reweighting methods, including PMI, tstats, chi-square, and our own method. We recommend to use chi-square, which is also the default choice in our code.
`--cut_off_k` can be used to adjust the threshold of ranking for reweighting, and the default value is 10000, which means we select the term pairs where the ranking of both terms is less than 10000, and the ranking of the term pair is also less than 10000.

## Session-level analysis
Session-level analysis contains session grouping, prompt frequency across users, edit distance within sessions, prompt repeats across session, edits analysis (added/deleted/replaced words of two adjacent prompts within sessions). See more details about sessions in our paper. Note that since sessions are grouped with a threshold of time, session-level analysis only supports `midjourney/diffusiondb`.

```python
python session_level.py --dataset_name <dataset_name> --data_path <the path of the tokenized file> --session_df_path <path of session file> --threshold <the threshold used for session file>
```

`--threshold` can be used to set the time threshold for sessions, and the default value is 30 (min).

## Prompt-level analysis
Prompt-level analysis contains basic statistics, DiffusionDB prompt frequency ranking plot, timestamp grouping plot, rating histogram (only for SAC), and the correlation of ratings and prompt lengths (only for SAC). For convenience, we use the files grouped by sessions for DiffusionDB prompt frequency ranking plot, but note that the analysis is not necessarily dependent on session analysis.

```python
python prompt_level.py --dataset_name <dataset_name> --data_path <the path of the tokenized file> --session_df_path <path of session file> --save_root <path of result data files> --threshold <the threshold used for session file>
```

`--topk_prompts` can be used to adjust the number of top frequent prompts saved in the file.


<!-- ## Cite -->

## Contact
Feel free to contact us via email or submit new issues if you have problems.