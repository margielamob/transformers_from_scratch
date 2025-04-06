"""
Main script for training and evaluating machine translation models
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
import os
import sys

from model import TranslationRNN, TranslationTransformer
from utils import load_tokenizers, build_datasets, generate_batch, beam_search, greedy_search
from train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a machine translation model')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['rnn', 'gru', 'transformer'], 
                        help='Type of model to train')
    parser.add_argument('--dim_embedding', type=int, default=256, 
                        help='Dimension of token embeddings')
    parser.add_argument('--dim_hidden', type=int, default=512, 
                        help='Dimension of hidden layers')
    parser.add_argument('--n_layers', type=int, default=3, 
                        help='Number of layers in encoder/decoder')
    parser.add_argument('--n_heads', type=int, default=8, 
                        help='Number of attention heads (only for transformer)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--clip', type=float, default=5.0, 
                        help='Gradient clipping')
    parser.add_argument('--max_seq_len', type=int, default=60, 
                        help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2, 
                        help='Minimum token frequency to include in vocabulary')
    
    # Logging and saving
    parser.add_argument('--log_every', type=int, default=50, 
                        help='Log training metrics every N batches')
    parser.add_argument('--save_dir', type=str, default='checkpoints', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--wandb', action='store_true', 
                        help='Enable logging with Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='machine-translation', 
                        help='W&B project name')
    parser.add_argument('--wandb_group', type=str, default=None, 
                        help='W&B group name')
    
    # Data options
    parser.add_argument('--data_path', type=str, default='data/fra-eng.txt', 
                        help='Path to translation data file')
    parser.add_argument('--test_size', type=float, default=0.1, 
                        help='Proportion of data to use for validation')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizers
    en_tokenizer, fr_tokenizer = load_tokenizers()
    if en_tokenizer is None or fr_tokenizer is None:
        sys.exit(1)
    
    # Load and preprocess data
    print("Loading dataset...")
    try:
        df = pd.read_csv(args.data_path, sep='\t', names=['english', 'french', 'attribution'])
        translation_pairs = [(en, fr) for en, fr in zip(df['english'], df['french'])]
        train_data, val_data = train_test_split(
            translation_pairs, test_size=args.test_size, random_state=args.seed
        )
        print(f"Training set: {len(train_data)} examples")
        print(f"Validation set: {len(val_data)} examples")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Build datasets
    print("Building datasets and vocabularies...")
    train_dataset, val_dataset = build_datasets(
        args.max_seq_len,
        args.min_freq,
        en_tokenizer,
        fr_tokenizer,
        train_data,
        val_data,
    )
    
    print(f"English vocabulary size: {len(train_dataset.en_vocab):,}")
    print(f"French vocabulary size: {len(train_dataset.fr_vocab):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: generate_batch(
            batch, train_dataset.en_vocab['<pad>'], train_dataset.fr_vocab['<pad>']
        )
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: generate_batch(
            batch, train_dataset.en_vocab['<pad>'], train_dataset.fr_vocab['<pad>']
        )
    )
    
    # Build model
    print(f"Creating {args.model_type.upper()} model...")
    if args.model_type.lower() in ['rnn', 'gru']:
        model = TranslationRNN(
            n_tokens_src=len(train_dataset.en_vocab),
            n_tokens_tgt=len(train_dataset.fr_vocab),
            dim_embedding=args.dim_embedding,
            dim_hidden=args.dim_hidden,
            n_layers=args.n_layers,
            dropout=args.dropout,
            src_pad_idx=train_dataset.en_vocab['<pad>'],
            tgt_pad_idx=train_dataset.fr_vocab['<pad>'],
            model_type=args.model_type.upper(),
        )
    else:  # transformer
        model = TranslationTransformer(
            n_tokens_src=len(train_dataset.en_vocab),
            n_tokens_tgt=len(train_dataset.fr_vocab),
            n_heads=args.n_heads,
            dim_embedding=args.dim_embedding,
            dim_hidden=args.dim_hidden,
            n_layers=args.n_layers,
            dropout=args.dropout,
            src_pad_idx=train_dataset.en_vocab['<pad>'],
            tgt_pad_idx=train_dataset.fr_vocab['<pad>'],
        )
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    
    # Create weight matrix for loss function to downweight <unk> tokens
    weight_classes = torch.ones(len(train_dataset.fr_vocab), dtype=torch.float)
    weight_classes[train_dataset.fr_vocab['<unk>']] = 0.1  # Lower importance of unknown words
    loss_fn = nn.CrossEntropyLoss(
        weight=weight_classes,
        ignore_index=train_dataset.fr_vocab['<pad>'],  # Ignore padding tokens
    )
    
    # Create training config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'clip': args.clip,
        'device': device,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'loss': loss_fn,
        'max_sequence_length': args.max_seq_len,
        'min_token_freq': args.min_freq,
        'src_vocab': train_dataset.en_vocab,
        'tgt_vocab': train_dataset.fr_vocab,
        'src_tokenizer': en_tokenizer,
        'tgt_tokenizer': fr_tokenizer,
        'src_pad_idx': train_dataset.en_vocab['<pad>'],
        'tgt_pad_idx': train_dataset.fr_vocab['<pad>'],
        'log_every': args.log_every,
    }
    
    # Initialize W&B if enabled
    if args.wandb:
        wandb_config = {
            'model_type': args.model_type,
            'dim_embedding': args.dim_embedding,
            'dim_hidden': args.dim_hidden,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads if args.model_type.lower() == 'transformer' else None,
            'dropout': args.dropout,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'clip': args.clip,
            'max_seq_len': args.max_seq_len,
            'min_freq': args.min_freq,
            'en_vocab_size': len(train_dataset.en_vocab),
            'fr_vocab_size': len(train_dataset.fr_vocab),
        }
        
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or args.model_type,
            config=wandb_config,
            save_code=True,
        )
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Train the model
    train_model(model, config)
    
    # Save the final model
    model_path = os.path.join(
        args.save_dir, 
        f"{args.model_type}_emb{args.dim_embedding}_hid{args.dim_hidden}_layers{args.n_layers}.pt"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'en_vocab': train_dataset.en_vocab,
        'fr_vocab': train_dataset.fr_vocab,
        'config': vars(args),
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Test with some examples
    print("\nTesting with some examples:")
    
    test_sentences = [
        "Hello, how are you?",
        "I would like to book a hotel room.",
        "The weather is nice today.",
        "Can you help me find the train station?",
        "What time is the next bus to Paris?"
    ]
    
    model.eval()
    for sentence in test_sentences:
        print(f"\nEnglish: {sentence}")
        
        # Beam search translation
        translations = beam_search(
            model, 
            sentence,
            train_dataset.en_vocab,
            train_dataset.fr_vocab,
            en_tokenizer,
            device,
            beam_width=5,
            max_target=5,
            max_sentence_length=args.max_seq_len
        )
        print("Beam search translations:")
        for i, (translation, prob) in enumerate(translations[:3]):
            print(f"  {i+1}. ({prob*100:.2f}%) {translation}")
        
        # Greedy search translation
        greedy_translation = greedy_search(
            model,
            sentence,
            train_dataset.en_vocab,
            train_dataset.fr_vocab,
            en_tokenizer,
            device,
            args.max_seq_len
        )
        print(f"Greedy search: {greedy_translation}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()