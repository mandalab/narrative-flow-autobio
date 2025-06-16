from pprint import pprint
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig
import pandas as pd
import numpy as np
import sys
import argparse
import os


def to_tokens_and_logprobs(model, tokenizer, device, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(model.device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def logprobs_sentence(sentence, context, model, tokenizer, device, DEBUG = False):

    logprob_full = to_tokens_and_logprobs(model, tokenizer, device, context+' '+sentence) # Changed to this to account for the stripped sentences now

    context_tokens = tokenizer(context, padding=True, return_tensors="pt").input_ids
    num_tokens = context_tokens.shape[1] - 1 # - 1 to account for the <eos> token
    if DEBUG:
        print(f"Context tokens and logprob : {to_tokens_and_logprobs(model, tokenizer, context)}")
        print(f"Full sentence: {context+sentence}")
        print(f"Number of tokens in context: {num_tokens}")
        print(f"Batch length: {len(logprob_full[0])}")
        print(f"Batch tuples: ", logprob_full)

    log_prob_sum = 0
    for i, (token, p) in enumerate(logprob_full[0]):
        if i < num_tokens:
            continue
        log_prob_sum += p
    return log_prob_sum

def sequentiality_seq_sentence_wordlevel(prompt, next_sentence, topic, tokenizer, model, device):
    context_l2 = logprobs_sentence(next_sentence, prompt, model, tokenizer, device, DEBUG=False)
    topic_l1 = logprobs_sentence(next_sentence, topic, model, tokenizer, device, DEBUG=False)

    score = -1/len(next_sentence.split(' '))*(topic_l1-context_l2)
    return score, topic_l1/len(next_sentence.split(' ')), context_l2/len(next_sentence.split(' '))

def sequentiality_seq_paragraph(paragraph, topic, tokenizer, model, device):
    l=''.join(paragraph.split('\n'))
    l = l.split(".")
    l.pop()
    logits_=[]
    topic_score_list=[]
    context_score_list=[]
    prompt=topic
    next_sentence=""
    for i in range(len(l)):
        if i>=1:
            prompt = prompt + ' ' + l[i-1].strip() + "."
        next_sentence = l[i] + "."
        next_sentence = next_sentence.strip()
        # Added these 2 strips to remove the empty spaces at the start or end

        score, topic_score, context_score = sequentiality_seq_sentence_wordlevel(prompt, next_sentence, topic, tokenizer, model, device)
        logits_.append(score)
        topic_score_list.append(topic_score)
        context_score_list.append(context_score)
        
    try:
        avg=sum(logits_)/len(logits_)
        avg_topic=sum(topic_score_list)/len(topic_score_list)
        avg_context=sum(context_score_list)/len(context_score_list)
    except:
        avg=0
        avg_topic=0
        avg_context=0
        print(f'Score 0 : {prompt + next_sentence}')

    return avg, avg_topic, avg_context

def compute_sequentiality(df, column, topic_col, tokenizer, model, device):
    df[f'{column}_seq'] = np.nan
    df[f'{column}_topic_seq'] = np.nan
    df[f'{column}_context_seq'] = np.nan
    for j in range(len(df)):
        l=(df[column].iloc[j])
        seq, topic_seq, contex_seq = sequentiality_seq_paragraph(l,df[topic_col].iloc[j], tokenizer, model, device)
        df.loc[j, f'{column}_seq'] = seq
        df.loc[j, f'{column}_topic_seq'] = topic_seq
        df.loc[j, f'{column}_context_seq'] = contex_seq
        # df[f'{column}_prob'].iloc[j] = similarity_score_paragraph(l,df['topic'].iloc[j], tokenizer, model, device)
        if j%25==0 or j==len(df)-1:
            print(f"Paragraph {j} done")
    return df

# Def main taking command line input for the run_name
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate sequentality for the given paragraphs.")
    parser.add_argument("--run_name", "-r", type=str, help="Name of the run to save the generated topics.")
    parser.add_argument("--dataset", "-d", type=str, help="Name of the dataset in the datsets directory.")
    parser.add_argument("--topic", "-t", type=str, help="Name of the column in the dataset to use as the topic.", default='topic')
    parser.add_argument("--column", "-c", type=str, help="Name of the column in the dataset to calculate sequentiality for.")
    parser.add_argument("--model", "-m", type=str, help="Name of the model to use for calculating sequentiality.", default='llama')
    args = parser.parse_args()

    df = pd.read_csv(f'../data/datasets/{args.dataset}.csv')
    print("Loaded existing DataFrame.")

    model_dict = {
        'llama': "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        'gemini': "hugging-quants/gemma-2-9b-it-AWQ-INT4",
        'qwen': "Qwen/Qwen2.5-7B-Instruct-AWQ",
        'falcon': "tiiuae/Falcon3-10B-Instruct-AWQ",
    }

    model_name_or_path = model_dict[args.model]

    print(f"Using model: {model_name_or_path}")

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # if scratch directory doesn't exist, create it
    os.makedirs("/scratch/autobio-cache/", exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map= DEVICE,
        cache_dir="/scratch/autobio-cache/",
    )
    model.config.pad_token_id = model.config.eos_token_id

    # Strip model name of special characters
    model_name = model_name_or_path.split("/")[-1]
    # Remove special characters from model name
    model_name = model_name.replace('/([^a-z0-9 ]+)/gi', '-')

    df = compute_sequentiality(df, args.column, args.topic, tokenizer, model, DEVICE)

    # Make directory if it doesn't exist
    os.makedirs(f'../data/scores/{args.run_name}', exist_ok=True)

    df.to_csv(f'../data/scores/{args.run_name}/{args.dataset}_scores_{model_name}.csv', index=False)
