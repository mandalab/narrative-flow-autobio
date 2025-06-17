# Overview

This file describes the functions, arguments, examples as well as the pre-requisites needed for each script in this folder. 

## create_autobio.py

This script iterates through pairs of autobiographies and biographies, matching paragraphs based on embedding similarity and rouge scores, creating a final dataset of matched paragraphs of autobiographies and biographies.

### Arguments

- `--run_name` (-r): name under which to save results under output CSV (and interim results in `../data/top_pairs/{run_name}`).  
- `--auto_dir` (-a): directory where the autobiographies are stored under in txt format.
- `--bio_dir` (-b): directory where the biographies are stored under in txt format.

### Usage 
```bash
python create_autobio.py \
  --run_name=myrun \
  --auto_dir=myautodir \
  --bio_dir=mybiodir
```

### Requirements

Both directories should contain the books in the format `auto_{author}.txt` or `bio_{author}.txt` with no underscores or dots in the authors name. Each corresponding autobiography must have a biography (and vice versa), under the same naming scheme to be matched. 


## generate_topics.py

This script generates topic labels for paragraphs (one-shot or zero-shot) in a DataFrame using a LLM pipeline. 

### Arguments

- `--run_name` (-r): name under which to save results under output CSV.  
- `--column`: column name for unpaired mode (only one paragraph) (default 'para'). 
- `--zero-shot`: to use zero-shot template or not (runs one-shot by default if not passed) 

### Usage 

#### one-shot topic generation
```bash
python generate_topics.py \
  --run_name=myrun \
  --column=paragraphs \
```

#### zero-shot topic generation
```bash
python generate_topics.py \
  --run_name=myrun \
  --column=paragraphs \
  --zero-shot
```

### Requirements

Dataset must contain `column` column of stories, with a row for each story.

## calculate_sequentiality

This script computes a “sequentiality” loss for paragraphs in a dataset using LLMs.

### Arguments:

- `run_name` (-r): name under which to save output CSV.

- `dataset (-d)`: CSV name (without .csv) for the dataset in ../data/datasets/ for which sequentiality is to be calculated.

- `topic (-t)`: column name for topic (default 'topic').

- `column (-c)`: column name for paragraphs to analyze.

- `model (-m)`: key in model_dict (default 'llama').


### Usage
```bash
python compute_sequentiality.py \
  --run_name=myrun \
  --dataset=my_datafile \
  --column=paragraph \
  --topic=topic \
  --model=llama \
```

### Requirements

Dataset must exist and have both a `topic` and `column` column, consisting of the stories in `column` and their corresponding topic in `topic`.

