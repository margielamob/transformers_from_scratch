# Neural Machine Translation from Scratch

This repository contains implementations of various Neural Machine Translation (NMT) models built from scratch with PyTorch. The models include:

1. **Vanilla RNN** - A basic recurrent neural network implementation
2. **GRU-RNN** - A Gated Recurrent Unit implementation
3. **Transformer** - A full encoder-decoder transformer architecture based on the "Attention is All You Need" paper

These models are trained for English to French translation using a dataset of sentence pairs. The implementations were first developed in the notebooks/transformer_charlm.ipynb notebook as part of a machine learning coursework. I then refactored the code into this more structured repository to showcase the work and create a foundation for future iterations and improvements.

## Project Structure

```
transformer-from-scratch/
├── notebooks/
│   └── transformer_charlm.ipynb  <- Original notebook with all implementations
├── src/
│   ├── model.py                  <- Model architecture implementations
│   ├── train.py                  <- Training utilities
│   ├── utils.py                  <- Data processing and evaluation utilities
│   └── main.py                   <- Main script for training and testing
├── README.md
└── LICENSE
```

## Requirements

- Python 3.8+
- PyTorch 2.1.2+
- torchtext 0.16.2+
- spaCy (with English and French models)
- pandas
- numpy
- scikit-learn
- wandb (optional, for experiment tracking)

You can install the necessary dependencies with:

```bash
pip install torch==2.1.2 torchtext==0.16.2 numpy==1.23.5 scipy==1.9.3 scikit-learn==1.1.3 pandas spacy einops wandb
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## Dataset

The models are trained on the [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/) dataset, specifically the English-French pairs. Download the dataset:

```bash
wget http://www.manythings.org/anki/fra-eng.zip
unzip fra-eng.zip
```

## Usage

### Training a Model

To train a model, run the `main.py` script with your desired parameters:

```bash
# Train a Transformer model
python src/main.py --model_type transformer --dim_embedding 256 --dim_hidden 512 --n_layers 3 --n_heads 8 --dropout 0.1 --epochs 5 --batch_size 128 --wandb

# Train a GRU-RNN model
python src/main.py --model_type gru --dim_embedding 256 --dim_hidden 512 --n_layers 2 --dropout 0.1 --epochs 5 --batch_size 128

# Train a vanilla RNN model
python src/main.py --model_type rnn --dim_embedding 256 --dim_hidden 512 --n_layers 2 --dropout 0.1 --epochs 5 --batch_size 128
```

### Command-line Arguments

- `--model_type`: Type of model to train (rnn, gru, transformer)
- `--dim_embedding`: Dimension of token embeddings
- `--dim_hidden`: Dimension of hidden layers
- `--n_layers`: Number of layers in encoder/decoder
- `--n_heads`: Number of attention heads (only for transformer)
- `--dropout`: Dropout rate
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--clip`: Gradient clipping
- `--max_seq_len`: Maximum sequence length
- `--min_freq`: Minimum token frequency for vocabulary
- `--log_every`: Log training metrics every N batches
- `--save_dir`: Directory to save model checkpoints
- `--wandb`: Enable logging with Weights & Biases
- `--wandb_project`: W&B project name
- `--wandb_group`: W&B group name
- `--data_path`: Path to translation data file
- `--test_size`: Proportion of data to use for validation
- `--seed`: Random seed

### Using a Trained Model

Load a trained model and translate new sentences:

```python
import torch
from src.model import TranslationTransformer
from src.utils import load_tokenizers, beam_search

# Load the saved model
checkpoint = torch.load('checkpoints/transformer_emb256_hid512_layers3.pt')
model_config = checkpoint['config']

# Initialize tokenizers
en_tokenizer, fr_tokenizer = load_tokenizers()

# Create model with same architecture
model = TranslationTransformer(
    n_tokens_src=len(checkpoint['en_vocab']),
    n_tokens_tgt=len(checkpoint['fr_vocab']),
    n_heads=model_config['n_heads'],
    dim_embedding=model_config['dim_embedding'],
    dim_hidden=model_config['dim_hidden'],
    n_layers=model_config['n_layers'],
    dropout=0.0,  # Set to 0 for inference
    src_pad_idx=checkpoint['en_vocab']['<pad>'],
    tgt_pad_idx=checkpoint['fr_vocab']['<pad>']
)

# Load saved weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Translate a sentence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentence = "Hello, how are you?"
translations = beam_search(
    model,
    sentence,
    checkpoint['en_vocab'],
    checkpoint['fr_vocab'],
    en_tokenizer,
    device,
    beam_width=5,
    max_target=5,
    max_sentence_length=model_config['max_seq_len']
)

for translation, prob in translations[:3]:
    print(f"({prob*100:.2f}%) {translation}")
```

## Implementation Details

The repository includes implementations of three different neural machine translation models:

### Vanilla RNN

The vanilla RNN uses a simple recurrent unit that updates its hidden state with a linear transformation and activation function. The encoder and decoder are separate RNN modules.

### GRU-RNN

The GRU-RNN uses Gated Recurrent Units, which have update and reset gates to control the flow of information. This allows the model to better capture long-term dependencies in the sequences.

### Transformer

The Transformer model uses self-attention and cross-attention mechanisms instead of recurrence. It processes all sequence positions in parallel and uses positional encodings to maintain sequence order information.

## Evaluation and Beam Search

The models are evaluated using:

1. Loss on the validation set
2. Top-1, Top-5, and Top-10 accuracy
3. Beam search for generating translations

Beam search keeps track of multiple hypotheses at each decoding step, which often yields better translations than greedy decoding.

## Future Work
JAX Implementation
One of the planned improvements is to rewrite this codebase using JAX for better performance, particularly to leverage Apple Silicon acceleration. JAX's just-in-time compilation and automatic differentiation should provide significant speed improvements over the current PyTorch implementation.

## Acknowledgments

- The Transformer implementation is based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- The dataset is from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/)