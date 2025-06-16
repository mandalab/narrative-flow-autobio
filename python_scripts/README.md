


## calculate_sequentiality

This script computes a “sequentiality” loss for paragraphs in a dataset using LLMs.


### Dependencies

// Amal u need to add here


### Arguments:

- `run_name` (-r): name under which to save output CSV.

- `dataset (-d)`: CSV name (without .csv) for the dataset in ../data/datasets/ for which sequentiality is to be calculated.

- `split`: optional; supports 'h1', 'h2', 'bio', 'auto', or empty. If 'h1'/'h2', uses first/second half of DataFrame/ or biography/autobiography for paragraph matched data.

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
  --split=h1

```


## generate_topics.py

This script generates topic labels for paragraphs (single or paired) in a DataFrame using a LLM pipeline. 

### Dependencies

### Arguments

- `--run_name` (-r): name under which to save output CSV.  
- `--gpu` (-g): GPU index (e.g., 0).  
- `--column`: column name for unpaired mode (only one paragraph) (default 'para').  
- `--column2`: second column name for paired mode (default 'para').  
- `--paired`: flag; if set, runs paired mode, else single paragraph.  
- `--save_modifier`: suffix for output CSV filename (default 'combined'). 

### Usage 

#### unpaired paragraph mode
```bash
python generate_topics.py --run_name=myrun --gpu=0 --column=paragraph_text
```

#### paired paragraph mode
```bash
python generate_topics.py --run_name=myrun --gpu=0 --paired --column=bio_para --column2=auto_para --save_modifier=paired
```


# Additional Notes AMAL
How to describe the run_name please re consider that.