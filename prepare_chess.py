import os
import re
from tqdm import tqdm
import numpy as np
import pickle
from datasets import load_dataset
import chess
import chess.pgn
import io
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

out_dir = './chess'
os.makedirs(out_dir, exist_ok=True)
num_proc = 64
num_threads = 64
seed = 42
train_size = None
test_size = 16384
batch_size = 1024

# Add FEN notation specific tokens
allowed_tokens = [
    ',', ';', '+', '#', '=', '?', '!', '\n', ' ', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-', 'x', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'K', 'Q', 'R', 'B', 'N', 'O', 'P', '(', ')',
    'l', 'k', 'v', ':', '*', '|', 'r', 'n', 'b', 'q', 'p', 'w', 'm'
]
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
allowed_tokens = [BOS_TOKEN, EOS_TOKEN] + allowed_tokens

# Create a mapping from tokens to integers
stoi = {token: i for i, token in enumerate(allowed_tokens)}
itos = {i: token for i, token in enumerate(allowed_tokens)}

# Get the BOS and EOS token IDs
BOS_TOKEN_ID = stoi[BOS_TOKEN]
EOS_TOKEN_ID = stoi[EOS_TOKEN]

# Pre-compile regular expressions
move_number_pattern = re.compile(r'\d+\.\.?\.?\s*')
curly_braces_pattern = re.compile(r'\{[^}]*\}')
whitespace_pattern = re.compile(r'\s+')

# Create a lookup set for allowed tokens
allowed_tokens_set = set(allowed_tokens)

def encode(text, add_bos=True, add_eos=True):
    # Optimize by pre-allocating array of correct size
    result = np.zeros(len(text) + (1 if add_bos else 0) + (1 if add_eos else 0), dtype=np.uint8)
    idx = 0
    
    if add_bos:
        result[idx] = BOS_TOKEN_ID
        idx += 1
    
    for char in text:
        result[idx] = stoi[char]
        idx += 1
    
    if add_eos:
        result[idx] = EOS_TOKEN_ID
        idx += 1
    
    return result[:idx]

@lru_cache(maxsize=100000)  # Increased cache size
def get_final_position(raw_movetext):
    try:
        # Create a minimal PGN
        pgn_text = "[Event \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n" + raw_movetext
        
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None
        
        # More efficient approach - use game.end() to get last position
        board = game.end().board()
        return board.fen()
    except Exception:
        return None

def fast_format_moves(moves):
    """Format moves with a single pass through the array"""
    if not moves:
        return ""
    
    # Pre-allocate result with estimated size (average move length ~3 chars + separator)
    result = []
    for i, move in enumerate(moves):
        # Append move with appropriate separator
        if i % 2 == 0:  # White's move
            result.append(move + ",")
        else:  # Black's move
            result.append(move + ";")
    
    return ''.join(result)

def process(example):
    # Extract what we need
    original_movetext = example['movetext']
    
    # Clean text with single regex pass
    cleaned_text = curly_braces_pattern.sub('', move_number_pattern.sub('', original_movetext))
    cleaned_text = whitespace_pattern.sub(' ', cleaned_text).strip()
    
    # Get moves and format in one step
    moves = cleaned_text.split()
    formatted_text = fast_format_moves(moves)
    
    # Get final board position
    # NOTE: This step takes a long time. You can skip it by setting final_position to an empty string.
    final_position = get_final_position(original_movetext) 
    # final_position = ""
    
    # Get Elo ratings
    white_elo = str(example['WhiteElo'])
    black_elo = str(example['BlackElo'])
    
    # Build combined text efficiently
    if final_position:
        combined_text = formatted_text + "|" + white_elo + "|" + black_elo + "|" + final_position
    else:
        combined_text = formatted_text
    
    # Encode efficiently
    ids = encode(combined_text, add_bos=True, add_eos=True)
    
    return {
        'ids': ids,
        'len': len(ids)
    }

def process_and_write_batch(batch, tokens_arr, start_idx):
    """Process a batch of examples and write results at once"""
    end_idx = start_idx
    all_tokens = []
    total_len = 0
    
    # Process all examples in the batch
    for example in batch:
        ids = np.array(example['ids'], dtype=np.uint8)
        all_tokens.append(ids)
        total_len += len(ids)
    
    # Flatten and write
    flat_tokens = np.concatenate(all_tokens, dtype=np.uint8)
    end_idx = start_idx + total_len
    tokens_arr[start_idx:end_idx] = flat_tokens
    
    return end_idx

if __name__ == '__main__':
    print(f"Using {num_proc} processes for dataset operations and {num_threads} threads for I/O")
    
    # Directly specify the parquet files to load
    base_url = "https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/data/year=2025/month=01/"
    data_files = {"train": [base_url + f"train-{i:05d}-of-00072.parquet" for i in range(36)]}
    
    # Load the dataset - use streaming for even less memory usage if possible
    dataset = load_dataset(
        "parquet", 
        data_files=data_files, 
        split=f"train[:{train_size}]" if train_size is not None else "train", 
        num_proc=num_proc
    )
    
    print(f"Total examples: {len(dataset)/1e6}M")
    
    # Process dataset with optimized map function
    processed_dataset = dataset.map(
        process,
        remove_columns=['movetext'],
        num_proc=num_proc,
        desc="processing dataset",
        batch_size=1000  # Use batching within map
    )
    
    # Split into train and test
    processed_dataset = processed_dataset.shuffle(seed=seed)
    if train_size is None:
        train_size = len(processed_dataset) - test_size
    
    splits = {
        'train': processed_dataset.select(range(train_size - test_size)),
        'test': processed_dataset.select(range(train_size - test_size, train_size))
    }
    
    # Save each split with highly optimized batch writing
    for split, ds in splits.items():
        tokens_filename = os.path.join(out_dir, f'{split}.bin')
        
        # Calculate total length of all sequences
        arr_len = np.sum(ds['len'], dtype=np.uint64)
        print(f'{split}: {arr_len/1e9}B tokens, {len(ds)/1e6}M examples')
        
        # Create memory map for the tokens
        tokens_dtype = np.uint8
        tokens_arr = np.memmap(tokens_filename, dtype=tokens_dtype, mode='w+', shape=(arr_len,))
        
        # Use larger batch size and process with threads
        token_idx = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Process in larger chunks for better throughput
            for i in tqdm(range(0, len(ds), batch_size), desc=f'writing {split} data'):
                batch = ds.select(range(i, min(i + batch_size, len(ds))))
                end_idx = process_and_write_batch(batch, tokens_arr, token_idx)
                token_idx = end_idx
        
        # Ensure data is written
        tokens_arr.flush()
    
    # Save metadata
    meta = {
        'vocab_size': len(allowed_tokens),
        'bos_token_id': BOS_TOKEN_ID,
        'eos_token_id': EOS_TOKEN_ID,
        'seed': seed,
        'train_size': train_size,
        'test_size': test_size,
        'format': 'moves|white_elo|black_elo|fen_position'
    }
    
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("Dataset preparation complete!")