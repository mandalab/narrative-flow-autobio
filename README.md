# From Stories to Statistics: Methodological Biases in LLM-Based Narrative Flow Quantification

Repository for the paper - "From Stories to Statistics: Methodological Biases in LLM-Based Narrative Flow Quantification" accepted at CoNLL'25.


## Setup

This code was run on `python==3.9.20`. Use pip to install the dependencies from the `requirements.txt` file as - 

```bash
pip install requirements.txt
```

## Datasets

The hippocorpus dataset is available at https://www.microsoft.com/en-us/download/details.aspx?id=105291. For the auto-bio dataset, due to copyright constraints we cannot make the books or the dataset public. However, we provide a pipeline to recreate our dataset in `python_scripts/create_autobio.py`, after acquring all the books listed in the paper (refer to `python_scripts/README.md` for more details).


## Workflow

To replicate the results of the paper, follow this general pipeline

1. Create the auto-bio paired dataset
2. Generate the topics (sampling randomly from either group) for the auto-bio dataset
3. Calculate sequentiality for the auto-bio dataset.
4. Analyze difference among the two groups.
5. Acquire the hippocorpus dataset and after filtering and merging, generate topics for both stories.
6. Calculate sequentiality under the different topic conditions.
7. Analyze the difference among the two groups.

For the analysis part, refer to `analyis/score_difference.ipynb` file for more details. For all the remaining parts, refer to `python_scripts/README.md` for further details about each script and what it entails. 
