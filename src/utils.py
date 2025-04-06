"""
Utilities for data loading, preprocessing, and tokenization for machine translation.
"""

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.vocab import build_vocab_from_iterator, Vocab
from itertools import takewhile
from collections import Counter, defaultdict
import spacy
from typing import List, Tuple, Dict, Callable, Any, Optional

# Special tokens
SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']

class TranslationDataset(Dataset):
    def __init__(
            self,
            dataset: list,
            en_vocab: Vocab,
            fr_vocab: Vocab,
            en_tokenizer,
            fr_tokenizer,
        ):
        super().__init__()

        self.dataset = dataset
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer

    def __len__(self):
        """Return the number of examples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        """Return a sample.

        Args
        ----
            index: Index of the sample.

        Output
        ------
            en_tokens: English tokens of the sample, as a LongTensor.
            fr_tokens: French tokens of the sample, as a LongTensor.
        """
        # Get the strings
        en_sentence, fr_sentence = self.dataset[index]

        # To list of words
        # We also add the beginning-of-sentence and end-of-sentence tokens
        en_tokens = ['<bos>'] + self.en_tokenizer(en_sentence) + ['<eos>']
        fr_tokens = ['<bos>'] + self.fr_tokenizer(fr_sentence) + ['<eos>']

        # To list of tokens
        en_tokens = self.en_vocab(en_tokens)  # list[int]
        fr_tokens = self.fr_vocab(fr_tokens)

        return torch.LongTensor(en_tokens), torch.LongTensor(fr_tokens)


def yield_tokens(dataset, tokenizer, lang):
    """Tokenize the whole dataset and yield the tokens.
    """
    assert lang in ('en', 'fr')
    sentence_idx = 0 if lang == 'en' else 1

    for sentences in dataset:
        sentence = sentences[sentence_idx]
        tokens = tokenizer(sentence)
        yield tokens


def build_vocab(dataset: list, en_tokenizer, fr_tokenizer, min_freq: int):
    """Return two vocabularies, one for each language.
    """
    en_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, en_tokenizer, 'en'),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    en_vocab.set_default_index(en_vocab['<unk>'])  # Default token for unknown words

    fr_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, fr_tokenizer, 'fr'),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    fr_vocab.set_default_index(fr_vocab['<unk>'])

    return en_vocab, fr_vocab


def preprocess(
        dataset: list,
        en_tokenizer,
        fr_tokenizer,
        max_words: int,
    ) -> list:
    """Preprocess the dataset.
    Remove samples where at least one of the sentences are too long.
    Those samples takes too much memory.
    Also remove the pending '\n' at the end of sentences.
    """
    filtered = []

    for en_s, fr_s in dataset:
        if len(en_tokenizer(en_s)) >= max_words or len(fr_tokenizer(fr_s)) >= max_words:
            continue

        en_s = en_s.replace('\n', '')
        fr_s = fr_s.replace('\n', '')

        filtered.append((en_s, fr_s))

    return filtered


def build_datasets(
        max_sequence_length: int,
        min_token_freq: int,
        en_tokenizer,
        fr_tokenizer,
        train: list,
        val: list,
    ) -> tuple:
    """Build the training, validation and testing datasets.
    It takes care of the vocabulary creation.

    Args
    ----
        - max_sequence_length: Maximum number of tokens in each sequences.
            Having big sequences increases dramatically the VRAM taken during training.
        - min_token_freq: Minimum number of occurences each token must have
            to be saved in the vocabulary. Reducing this number increases
            the vocabularies's size.
        - en_tokenizer: Tokenizer for the english sentences.
        - fr_tokenizer: Tokenizer for the french sentences.
        - train and val: List containing the pairs (english, french) sentences.


    Output
    ------
        - (train_dataset, val_dataset): Tuple of the two TranslationDataset objects.
    """
    datasets = [
        preprocess(samples, en_tokenizer, fr_tokenizer, max_sequence_length)
        for samples in [train, val]
    ]

    en_vocab, fr_vocab = build_vocab(datasets[0], en_tokenizer, fr_tokenizer, min_token_freq)

    datasets = [
        TranslationDataset(samples, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)
        for samples in datasets
    ]

    return datasets


def generate_batch(data_batch: list, src_pad_idx: int, tgt_pad_idx: int) -> tuple:
    """Add padding to the given batch so that all
    the samples are of the same size.

    Args
    ----
        data_batch: List of samples.
            Each sample is a tuple of LongTensors of varying size.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.

    Output
    ------
        en_batch: Batch of tokens for the padded english sentences.
            Shape of [batch_size, max_en_len].
        fr_batch: Batch of tokens for the padded french sentences.
            Shape of [batch_size, max_fr_len].
    """
    en_batch, fr_batch = [], []
    for en_tokens, fr_tokens in data_batch:
        en_batch.append(en_tokens)
        fr_batch.append(fr_tokens)

    en_batch = pad_sequence(en_batch, padding_value=src_pad_idx, batch_first=True)
    fr_batch = pad_sequence(fr_batch, padding_value=tgt_pad_idx, batch_first=True)
    return en_batch, fr_batch


def beautify(sentence: str) -> str:
    """Removes useless spaces.
    """
    punc = {'.', ',', ';'}
    for p in punc:
        sentence = sentence.replace(f' {p}', p)

    links = {'-', "'"}
    for l in links:
        sentence = sentence.replace(f'{l} ', l)
        sentence = sentence.replace(f' {l}', l)

    return sentence


def beam_search(
        model: nn.Module,
        source: str,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        src_tokenizer,
        device: str,
        beam_width: int,
        max_target: int,
        max_sentence_length: int,
    ) -> list:
    """Do a beam search to produce probable translations.

    Args
    ----
        model: The translation model. Assumes it produces linear score (before softmax).
        source: The sentence to translate.
        src_vocab: The source vocabulary.
        tgt_vocab: The target vocabulary.
        device: Device to which we make the inference.
        beam_width: Number of top-k tokens we keep at each stage.
        max_target: Maximum number of target sentences we keep at the end of each stage.
        max_sentence_length: Maximum number of tokens for the translated sentence.

    Output
    ------
        sentences: List of sentences orderer by their likelihood.
    """
    src_tokens = ['<bos>'] + src_tokenizer(source) + ['<eos>']
    src_tokens = src_vocab(src_tokens)

    tgt_tokens = ['<bos>']
    tgt_tokens = tgt_vocab(tgt_tokens)

    # To tensor and add unitary batch dimension
    src_tokens = torch.LongTensor(src_tokens).to(device)
    tgt_tokens = torch.LongTensor(tgt_tokens).unsqueeze(dim=0).to(device)
    target_probs = torch.FloatTensor([1]).to(device)
    model.to(device)

    EOS_IDX = tgt_vocab['<eos>']
    with torch.no_grad():
        while tgt_tokens.shape[1] < max_sentence_length:
            batch_size, n_tokens = tgt_tokens.shape

            # Get next beams
            src = torch.repeat_interleave(src_tokens.unsqueeze(0), batch_size, dim=0)
            predicted = model.forward(src, tgt_tokens)
            predicted = torch.softmax(predicted, dim=-1)
            probs, predicted = predicted[:, -1].topk(k=beam_width, dim=-1)

            # Separate between terminated sentences and the others
            idx_terminated = torch.LongTensor([i for i, t in enumerate(tgt_tokens) if EOS_IDX in t]).to(device)
            idx_not_terminated = torch.LongTensor([i for i, t in enumerate(tgt_tokens) if EOS_IDX not in t]).to(device)

            tgt_terminated = torch.index_select(tgt_tokens, dim=0, index=idx_terminated)
            tgt_probs_terminated = torch.index_select(target_probs, dim=0, index=idx_terminated)

            filter_t = lambda t: torch.index_select(t, dim=0, index=idx_not_terminated)
            tgt_others = filter_t(tgt_tokens)
            tgt_probs_others = filter_t(target_probs)
            predicted = filter_t(predicted)
            probs = filter_t(probs)

            # Add the top tokens to the previous target sentences
            batch_size, n_beams = predicted.shape
            n_tokens = tgt_others.shape[1]
            
            # Repeat each sentence n_beams times
            tgt_others = torch.repeat_interleave(tgt_others, n_beams, dim=0)
            
            # Add the predicted tokens
            beams = predicted.view(-1, 1)
            tgt_others = torch.cat((tgt_others, beams), dim=1)

            # Add padding to terminated target
            if len(tgt_terminated) > 0:
                padd = torch.zeros((len(tgt_terminated), 1), dtype=torch.long, device=device)
                tgt_terminated = torch.cat(
                    (tgt_terminated, padd),
                    dim=1
                )

            # Update each target sentence probabilities
            tgt_probs_others = torch.repeat_interleave(tgt_probs_others, beam_width)
            tgt_probs_others *= probs.flatten()
            
            if len(tgt_probs_terminated) > 0:
                tgt_probs_terminated *= 0.999  # Penalize short sequences overtime

            # Group up the terminated and the others
            target_probs = torch.cat(
                (tgt_probs_others, tgt_probs_terminated),
                dim=0
            ) if len(tgt_probs_terminated) > 0 else tgt_probs_others
            
            tgt_tokens = torch.cat(
                (tgt_others, tgt_terminated),
                dim=0
            ) if len(tgt_terminated) > 0 else tgt_others

            # Keep only the top `max_target` target sentences
            if target_probs.shape[0] <= max_target:
                continue

            target_probs, indices = target_probs.topk(k=max_target, dim=0)
            tgt_tokens = torch.index_select(tgt_tokens, dim=0, index=indices)

    sentences = []
    for tgt_sentence in tgt_tokens:
        tgt_sentence = list(tgt_sentence)[1:]  # Remove <bos> token
        tgt_sentence = list(takewhile(lambda t: t != EOS_IDX, tgt_sentence))
        tgt_sentence = ' '.join(tgt_vocab.lookup_tokens(tgt_sentence))
        sentences.append(tgt_sentence)

    sentences = [beautify(s) for s in sentences]

    # Join the sentences with their likelihood
    sentences = [(s, p.item()) for s, p in zip(sentences, target_probs)]
    # Sort the sentences by their likelihood
    sentences = [(s, p) for s, p in sorted(sentences, key=lambda k: k[1], reverse=True)]

    return sentences


def greedy_search(
        model: nn.Module,
        source: str,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        src_tokenizer,
        device: str,
        max_sentence_length: int,
    ) -> str:
    """Do a greedy search to produce translations.

    Args
    ----
        model: The translation model. Assumes it produces logits score (before softmax).
        source: The sentence to translate.
        src_vocab: The source vocabulary.
        tgt_vocab: The target vocabulary.
        device: Device to which we make the inference.
        max_sentence_length: Maximum number of tokens for the translated sentence.

    Output
    ------
        sentence: The translated source sentence.
    """
    src_tokens = src_tokenizer(source)
    src_indices = [src_vocab.stoi.get(token, src_vocab.stoi['<unk>']) for token
                   in src_tokens]
    src_indices = [src_vocab.stoi['<bos>']] + src_indices + [src_vocab.stoi['<eos>']]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(dim=0).to(device)
    tgt_indices = [tgt_vocab.stoi['<bos>']]

    # For TranslationRNN, we may want to get encoder output separately
    if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        with torch.no_grad():
            # Assume an encoder-decoder implementation
            src_emb = model.src_embeddings(src_tensor)
            encoder_output, hidden = model.encoder(src_emb)
            hidden = model.norm(hidden)
            
            for i in range(max_sentence_length):
                tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(dim=0).to(device)
                tgt_emb = model.tgt_embeddings(tgt_tensor)
                
                with torch.no_grad():
                    output, _ = model.decoder(tgt_emb, hidden)
                    next_token_logits = model.out_layer(output[:, -1, :])
                    next_token_index = next_token_logits.argmax(dim=-1).item()
                
                tgt_indices.append(next_token_index)
                if next_token_index == tgt_vocab.stoi['<eos>']:
                    break
    else:
        # General approach that should work with any model that takes source and target
        for i in range(max_sentence_length):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(dim=0).to(device)
            
            with torch.no_grad():
                output = model(src_tensor, tgt_tensor)
                next_token_logits = output[:, -1, :]
                next_token_index = next_token_logits.argmax(dim=-1).item()
            
            tgt_indices.append(next_token_index)
            if next_token_index == tgt_vocab.stoi['<eos>']:
                break

    result_tokens = [tgt_vocab.itos[idx] for idx in tgt_indices[1:]]  # Skip the <bos> token

    if result_tokens and result_tokens[-1] == '<eos>':
        result_tokens = result_tokens[:-1]  # Remove the <eos> token

    translated_sentence = ' '.join(result_tokens)
    return beautify(translated_sentence)


def topk_accuracy(
        real_tokens: torch.FloatTensor,
        probs_tokens: torch.FloatTensor,
        k: int,
        tgt_pad_idx: int,
    ) -> torch.FloatTensor:
    """Compute the top-k accuracy.
    We ignore the PAD tokens.

    Args
    ----
        real_tokens: Real tokens of the target sentence.
            Shape of [batch_size * n_tokens].
        probs_tokens: Tokens probability predicted by the model.
            Shape of [batch_size * n_tokens, n_target_vocabulary].
        k: Top-k accuracy threshold.
        tgt_pad_idx: Source padding index value.

    Output
    ------
        acc: Scalar top-k accuracy value.
    """
    total = (real_tokens != tgt_pad_idx).sum()

    _, pred_tokens = probs_tokens.topk(k=k, dim=-1)  # [batch_size * n_tokens, k]
    real_tokens = torch.repeat_interleave(real_tokens.unsqueeze(-1), k, dim=-1)  # [batch_size * n_tokens, k]

    good = (pred_tokens == real_tokens) & (real_tokens != tgt_pad_idx)
    acc = good.any(dim=-1).sum() / total
    return acc


def load_tokenizers():
    """Load the English and French tokenizers."""
    try:
        en_nlp = spacy.load('en_core_web_sm')
        fr_nlp = spacy.load('fr_core_news_sm')
        
        def en_tokenizer(text):
            return [tok.text.lower() for tok in en_nlp.tokenizer(text)]
        
        def fr_tokenizer(text):
            return [tok.text.lower() for tok in fr_nlp.tokenizer(text)]
        
        return en_tokenizer, fr_tokenizer
    except OSError:
        print("Error: Spacy models need to be downloaded. Please run:")
        print("python -m spacy download en_core_web_sm")
        print("python -m spacy download fr_core_news_sm")
        return None, None