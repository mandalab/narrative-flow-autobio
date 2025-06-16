from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
from IPython.display import clear_output
import argparse
import os


# Load the books
def load_book(file_path):
    """Load text from a book file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Split text into paragraphs
def split_into_paragraphs(text):
    """Split text into paragraphs based on double newlines."""
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if len(p.strip().split(' ')) > 20]
    return paragraphs

# Compute similarities between all pairs of paragraphs
def compute_similarities(auto_book_embeddings, bio_book_embeddings, auto_book_paragraphs, bio_book_paragraphs, similarity_threshold=0.7, DEBUG = False):
    """Compute cosine similarities between all pairs of paragraph embeddings."""
    similar_pairs = []

    temp_sim = cosine_similarity(auto_book_embeddings, bio_book_embeddings) # Computes NxM similarity matrix of paragraphs across two books (N paragraphs in book 1, M paragraphs in book 2)
    # Thresholding is done here to remove low similarity pairs and subsequently select the minimum similarity pair, change later if needed
    temp_sim[temp_sim < similarity_threshold] = 0

    for i, row in enumerate(temp_sim):
        if np.max(row) != 0:
            similar_pairs.append((auto_book_paragraphs[i], bio_book_paragraphs[np.argmax(row)], np.max(row)))

            if DEBUG:
                print(i, np.max(row), np.argmax(row))
                print('Book 1: ',auto_book_paragraphs[i])
                print('Book 2: ',bio_book_paragraphs[np.argmax(row)])
                print('---')

    return similar_pairs

def compute_embeddings(paragraphs, model):
    pool = model.start_multi_process_pool()

    embeddings = model.encode_multi_process(paragraphs, pool)

    model.stop_multi_process_pool(pool)

    return embeddings

def process_books(auto, bio, model, thresholds, run_name, output_file, upper_bound = 0.90, DEBUG = False, batch_size = 500):
    """Process two books and find similar paragraphs."""
    auto_book = load_book(auto)
    bio_book = load_book(bio)

    auto_book_paragraphs = split_into_paragraphs(auto_book)
    bio_book_paragraphs = split_into_paragraphs(bio_book)

    # If either book is empty, return
    if len(auto_book_paragraphs) == 0 or len(bio_book_paragraphs) == 0:
        return False

    print(len(auto_book_paragraphs))
    print(len(bio_book_paragraphs))

    # Split the paragraphs into batches of 500 and compute embeddings and concatenate the batches
    # Was OOM even with 3GPUs without batch size for inputs of 1000 paragraphs
    auto_book_embeddings = []
    bio_book_embeddings = []

    for i in range(0, len(auto_book_paragraphs), batch_size):
        auto_batch = auto_book_paragraphs[i:i+batch_size]
        auto_batch_embeddings = compute_embeddings(auto_batch, model)
        auto_book_embeddings.extend(auto_batch_embeddings)

    for i in range(0, len(bio_book_paragraphs), batch_size):
        bio_batch = bio_book_paragraphs[i:i+batch_size]
        bio_batch_embeddings = compute_embeddings(bio_batch, model)
        bio_book_embeddings.extend(bio_batch_embeddings)
    
    auto_book_embeddings = np.array(auto_book_embeddings)
    bio_book_embeddings = np.array(bio_book_embeddings)

    # Compute and display similar pairs
    similar_pairs = compute_similarities(auto_book_embeddings, bio_book_embeddings, auto_book_paragraphs, bio_book_paragraphs, similarity_threshold=thresholds[0])

    for threshold in thresholds:
        data = []
        output_file_path = f'../data/top_pairs/{run_name}/{output_file}_{threshold:.2f}.txt'  # Change this to your desired file path
        count = 0
        # To filter out text pairs within each threshold window
        upper_bound = threshold + 0.05 
        if threshold == 0.80:
            upper_bound = 0.90 # Edge case for last threshold
        # Display similar pairs
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for idx, (para1, para2, score) in enumerate(similar_pairs):
                if len(para1.split()) < 20 or len(para2.split()) < 20:
                    continue
                if score < threshold:
                    continue
                if score > upper_bound:
                    continue
                file.write(f"Pair {idx + 1}:\n")
                file.write(f"Auto Paragraph: {para1}\n")
                file.write(f"Bio Paragraph: {para2}\n")
                file.write(f"Similarity Score: {score:.2f}\n")
                file.write(f"Rouge Score: {rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(para1, para2)['rougeL'].fmeasure:.2f}\n")
                data.append((para1, para2, score, rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(para1, para2)['rougeL'].fmeasure))
                file.write("\n")
                count += 1

        df = pd.DataFrame(data, columns=['auto_para', 'bio_para', 'similarity_score', 'rouge_score'])
        df.to_csv(f'../data/top_pairs/{run_name}/{output_file}_{threshold:.2f}.csv', index=False)
        print(f"Number of similar pairs with threshold {threshold}: {count}")
    
    return True

def iterate_books(model, thresholds, run_name, upper_bound = 0.90, DEBUG = False, auto_dir = '../data/txt/auto/', bio_dir = '../data/txt/bio/'):
    # Iterate through all files in the auto directory
    for file1 in os.listdir(auto_dir):
        # Extract the author name from the file name
        author = file1.split('.')[0].split('_')[1]
        bio_file = f'{bio_dir}bio_{author}.txt'

        if not os.path.exists(bio_file):
            continue

        print(f'Processing {author}...')
        file1 = f'{auto_dir}{file1}'
        file2 = bio_file

        save_name = run_name+'_'+author
        # Make directory if it doesnt exist
        if not os.path.exists(f'../data/top_pairs/{run_name}'):
            os.makedirs(f'../data/top_pairs/{run_name}')

        # Check if the csv file already exists
        if os.path.exists(f'../data/top_pairs/{run_name}/{save_name}.csv'):
            print(f'{author} already processed...')
            continue

        success = process_books(file1, file2, model, thresholds, run_name, save_name, upper_bound, DEBUG)

        if not success:
            print(f'Failed to process {author}... No paragraphs found in one of the books')
            continue
        # Read all the csv files and combine them
        df = pd.DataFrame()
        for threshold in thresholds:
            df = pd.concat([df, pd.read_csv(f'../data/top_pairs/{run_name}/{save_name}_{threshold:.2f}.csv')], ignore_index=True)
        
        # Sort the dataframe by similarity score
        df = df.sort_values(by='similarity_score', ascending=False)
        df.to_csv(f'../data/top_pairs/{run_name}/{save_name}.csv', index=False)

# Def main taking command line input for the run_name
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process books to find similar paragraphs.")
    parser.add_argument("--run_name", "-r", type=str, help="Name of the run to save the paragraphs and dataset in.", default='runV3')
    parser.add_argument("--auto_dir", "-a", type=str, help="Directory containing auto books.", default='../data/txt_v3/auto/')
    parser.add_argument("--bio_dir", "-b", type=str, help="Directory containing bio books.", default='../data/txt_v3/bio/')
    args = parser.parse_args()

    model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
    thresholds = [0.65, 0.70, 0.75, 0.80]

    run_name = args.run_name
    auto_dir = args.auto_dir
    bio_dir = args.bio_dir

    iterate_books(model, thresholds, run_name, 1.0, DEBUG=False, auto_dir=auto_dir, bio_dir=bio_dir)
    combined_df = pd.DataFrame()

    # Iterate through all files in the auto directory
    for file1 in os.listdir(auto_dir):
        # Extract the author name from the file name
        author = file1.split('.')[0].split('_')[1]
        bio_file = f'{bio_dir}bio_{author}.txt'

        if not os.path.exists(bio_file):
            continue

        print(f'Processing {author}...')
        file1 = f'{auto_dir}{file1}'
        file2 = bio_file

        save_name = run_name+'_'+author
        try:
            df = pd.read_csv(f'../data/top_pairs/{run_name}/{save_name}.csv')
        except:
            continue

        # Add author name as a column to each df
        df['person'] = author

        # Combine all the dataframes
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Filter the combined dataframe based on rouge and similarity scores
        combined_df = combined_df[combined_df['rouge_score'] < 0.4]
        combined_df = combined_df[combined_df['similarity_score'] > 0.7]

        combined_df.to_csv(f'../data/datasets/{run_name}_pairs_dataset.csv', index=False)

