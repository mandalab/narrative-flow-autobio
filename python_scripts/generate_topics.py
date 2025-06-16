from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
import torch
import pandas as pd
import random
import sys
import argparse

def generate_topic_single_para(model, tokenizer, run_name, column, df, save_modifier='combined', template=None):
    prompt_template = """<s>[INST] A topic is a subject or theme that is discussed, written about, or explored in a piece of writing or in a conversation. It serves as the main idea or focus of a discussion or text. Identify the topic for the given paragraph. Return a single topic which is most relevent for the paragraph. Return only the topic in 1-2 sentences with no additional text or information.
        For example:

      "Paragraph: The man who explained the difference between net and gross to me, and much more besides, was Bruno Silva. Bruno would be our salvation at Groningen in those first few months. I remembered him as a Uruguayan international and as a player for Danubio – one of the third teams in Uruguay along with Defensor, behind Nacional and Peñarol. We used to get together to watch games from the Uruguayan league or we would meet up for family barbecues. We couldn’t find any Uruguayan steak so we managed to get hold of some Brazilian meat instead from a Brazilian who had played for many years in Groningen called Hugo Alves Velame. He was coaching in the academy at that point and he was someone else who was great with Sofi and me, becoming our translator whenever we had to deal with the club." [/INST]

      Topic: Bruno Silva's vital support in Groningen. </s>


       [INST] Identify the topic for the given paragraph. Return a single topic which is most relevent for the paragraph. Return only the topic in 1-2 sentences with no additional text or information.
      "Paragraph: {p1}" [/INST]
      """
    if template=='zero_shot':
        prompt_template = """<s>[INST] You are given a story or event that has happened in the given paragraph. Come up with a short summary of the event (2-3 sentences), written with enough details that you will remember what you wrote about in the future. Return only the summary in the format of "Topic: <summary>" with no additional text or information.
        "Paragraph: {p1}" </s>[/INST]
        """

    generation_params = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_params
    )

    df['topic'] = None

    save_file_name = f'../data/datasets/{run_name}_{save_modifier}.csv'

    try:
        new_df = pd.read_csv(save_file_name)
        print("Loaded existing DataFrame.")
    except:
        new_df = pd.DataFrame(columns=df.columns)
        print("Creating a new DataFrame.")

    def generate_topic(para):
        prompt = prompt_template.format(p1=para)
        # tokens = tokenizer(
        #     prompt,
        #     return_tensors='pt'
        # ).input_ids.cuda()

        pipe_output = pipe(prompt)[0]['generated_text']
        topic = pipe_output.split("Topic: ")[-1].split("</s>")[0].strip()
        return topic

    # Iterate through the rows of the original DataFrame
    for index, row in df.iterrows():
        para = row[column]
        
        # Read the saved dataframe and continue if the row is already processed
        # if para in new_df[column].values:
        #     print(f"Skipping row {index+1}, topic already generated.")
        #     continue

        topic = generate_topic(para)

        # Append the row and the generated topic to the new DataFrame
        new_df.loc[index] = row
        new_df.loc[index, 'topic'] = topic

        # Save the updated DataFrame after each iteration
        new_df.to_csv(save_file_name, index=False)
        print(f"Processed row {index+1}, topic generated and saved.")


# Def main taking command line input for the run_name
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate topics for the given paragraphs.")
    parser.add_argument("--run_name", "-r", type=str, help="Name of the run to save the generated topics.")
    parser.add_argument("--column", type=str, help="Column name to use for the generation.", default='para')
    parser.add_argument("--zero_shot", action='store_true', help="Use zero-shot template for topic generation.")
    args = parser.parse_args()

    model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=DEVICE,
    )

    df = pd.read_csv(f'../data/datasets/{args.run_name}.csv')

    if args.zero_shot:
        print('Using zero-shot template for topic generation.')
        generate_topic_single_para(model, tokenizer, args.run_name, args.column, df, save_modifier='zero_shot', template='zero_shot')
    else:
        print('Using default template for topic generation.')
        generate_topic_single_para(model, tokenizer, args.run_name, args.column, df, save_modifier='one_shot')

# Run the script with the following command
# python generate_topics.py --run_name=run1 --gpu=0 --paired
