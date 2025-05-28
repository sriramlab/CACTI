import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_KERAS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from os import path as ospath
from pandas import read_csv as pdreader
import torch
import numpy as np

# Import necessary libraries based on mode
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='Script to pre-compute column name/description embeddings.')
parser.add_argument('--data_file', type=str, default=None, help='Input data file with column names.') # prefer using desc_file instead of data_file
parser.add_argument('--desc_file', type=str, default=None, help='Input data file with column names and description. NOTE: col order needs to be aligned with data file.')
parser.add_argument('--embed_model', type=str, required=True, help='The model used to generate the embeddings.')  # Preferred: "bert-large-cased" or "Alibaba-NLP/gte-en-mlm-large" or "emilyalsentzer/Bio_ClinicalBERT"
parser.add_argument('--tkn_model', type=str, default=None, help='The tokenizer model (only used in BERT mode).')
parser.add_argument('--out', type=str, required=True, help='Dir to save the embedding results.')
parser.add_argument('--fname', type=str, required=True, help='Output file name.')
parser.add_argument('--mode', type=str, required=True, choices=['BERT', 'SentenceTransformer'], help='Embedding mode: BERT or SentenceTransformer')

#### Helpers
def load_colinfo(args):
    # Extract column names from original data
    if args.data_file is not None:
        # Read header from data file
        with open(args.data_file, 'r') as file:
            first_line = file.readline().strip()

        # Get list of column names from TSV or CSV
        colname_list = first_line.split('\t' if args.data_file.endswith('.tsv') else ',')
        colinfo_list = colname_list
    else:
        # Extract column names and info from description file
        # Assuming header present
        tmpf = pdreader(args.desc_file, sep='\t' if args.desc_file.endswith('.tsv') else ',')

        # Assuming the following data structure: colname (matching datafile), column description
        colname_list = tmpf.iloc[:, 0].tolist()
        colinfo_list = tmpf.iloc[:, 1].tolist()
        for i, col in enumerate(colname_list):
            cdef = col if colinfo_list[i] != colinfo_list[i] else colinfo_list[i]
            colinfo_list[i] = f"{col} (is defined as) : {cdef}"
            ### DEBUG
            print(colinfo_list[i])
            ###

    # Type check
    if not isinstance(colname_list, list) or not all(isinstance(item, str) for item in colname_list):
        raise TypeError("colname_list must be a list of strings")

    return colname_list, colinfo_list

def compute_embeddings(model, tokenizer, colnames, colinfo, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    special_tokens = ["FID", "IID", "ID"]

    if mode == 'BERT':
        model.to(device)
        model.eval()
        embeddings = []
        with torch.no_grad():
            for i, colname in enumerate(colnames):
                if colname.upper() in special_tokens:
                    # For special tokens, use a vector of zeros
                    embedding = torch.zeros(model.config.hidden_size)
                else:
                    tokens = tokenizer(
                        colinfo[i],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                        add_special_tokens=False
                    )
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    outputs = model(**tokens)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
                embeddings.append(embedding.numpy())
        embeddings = np.array(embeddings)
    
    # elif mode == 'SentenceTransformer':
    #     indices_to_encode = []
    #     sentences_to_encode = []
    #     for i, colname in enumerate(colnames):
    #         if colname.upper() not in special_tokens:
    #             indices_to_encode.append(i)
    #             sentences_to_encode.append(colinfo[i])

    #     embedding_dim = model.get_sentence_embedding_dimension()
    #     embeddings = np.zeros((len(colnames), embedding_dim))

    #     if sentences_to_encode:
    #         embeddings_encoded = model.encode(sentences_to_encode, convert_to_numpy=True)
    #         for idx, emb in zip(indices_to_encode, embeddings_encoded):
    #             embeddings[idx] = emb
        
    elif mode == 'SentenceTransformer': # batched version
        embedding_dim = model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(colnames), embedding_dim))

        # Process in batches
        batch_size = 16  # Adjust this based on your GPU memory
        for i in range(0, len(colnames), batch_size):
            print(f"SentenceTransformer running in batch for cols {i} - {i+batch_size}")
            batch_colnames = colnames[i:i+batch_size]
            batch_colinfo = colinfo[i:i+batch_size]

            sentences_to_encode = []
            indices_to_encode = []

            for j, colname in enumerate(batch_colnames):
                if colname.upper() not in special_tokens:
                    indices_to_encode.append(i + j)
                    sentences_to_encode.append(batch_colinfo[j])

            if sentences_to_encode:
                # Use model.encode() method which is optimized for batches
                batch_embeddings = model.encode(sentences_to_encode, 
                                                device=device, 
                                                show_progress_bar=False, 
                                                convert_to_numpy=True,
                                                normalize_embeddings=True)  # Normalizing can improve quality

                for idx, emb in zip(indices_to_encode, batch_embeddings):
                    embeddings[idx] = emb

            # Clear GPU cache after each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return embeddings

####

if __name__ == "__main__":
    args = parser.parse_args()

    if args.data_file is None and args.desc_file is None:
        raise ValueError('Need either --data_file OR --desc_file')

    if args.data_file is not None and args.desc_file is not None:
        raise ValueError('Conflict detected! \nCannot use both --data_file AND --desc_file; please pick one.')

    # Get column names/info
    colnames, colinfo = load_colinfo(args)

    # Initialize models based on mode
    if args.mode == 'BERT':
        # Initialize tokenizer and embedding model
        if args.tkn_model is None:
            tokenizer = AutoTokenizer.from_pretrained(args.embed_model,trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tkn_model)
        embedding_model = AutoModel.from_pretrained(args.embed_model,trust_remote_code=True)
    elif args.mode == 'SentenceTransformer':
        # Initialize the SentenceTransformer model
        tokenizer = None  # Not needed in this mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = SentenceTransformer(args.embed_model, device=device, trust_remote_code=True)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        print(f"Model embedding dimension: {embedding_dim}")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # Compute embeddings
    embeddings = compute_embeddings(embedding_model, tokenizer, colnames, colinfo, args.mode)
    print("Shape of embeddings:", embeddings.shape)

    # Save embeddings
    output_path = ospath.join(args.out, f"{args.fname}.npz")
    np.savez(output_path, colembed=embeddings, colnames=colnames)
    print(f"Embeddings saved to {output_path}")

    print("Embedding computation and saving completed successfully.")
