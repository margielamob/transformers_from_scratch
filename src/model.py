"""
Model architectures for machine translation:
- RNN-based models (Vanilla RNN and GRU-RNN)
- Transformer-based models
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################
# RNN Models
###################

class RNNCell(nn.Module):
    """A single RNN layer.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        dropout: Dropout rate.

    Important note: This layer does not exactly the same thing as nn.RNNCell does.
    PyTorch implementation is only doing one simple pass over one token for each batch.
    This implementation is taking the whole sequence of each batch and provide the
    final hidden state along with the embeddings of each token in each sequence.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            dropout: float,
        ):
        super().__init__()

        self.hidden_size = hidden_size

        # See pytorch definition: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.Wih = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.Whh = nn.Linear(hidden_size, hidden_size, device=DEVICE)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.Tanh()

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor) -> tuple:
        """Go through all the sequence in x, iteratively updating
        the hidden state h.

        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state.
                Shape of [batch_size, hidden_size].

        Output
        ------
            y: Token embeddings.
                Shape of [batch_size, seq_len, hidden_size].
            h: Last hidden state.
                Shape of [batch_size, hidden_size].
        """
        batch_size, seq_len, input_size = x.shape
        y = torch.zeros([batch_size, seq_len, self.hidden_size], device=DEVICE)

        for t in range(seq_len):
          input = x[:, t, :]
          w_input = self.Wih(input)
          w_hidden = self.Whh(h)
          h = self.act(w_input + w_hidden)
          y[:, t, :] = self.dropout(h)

        return y, h


class RNN(nn.Module):
    """Implementation of an RNN based
    on https://pytorch.org/docs/stable/generated/torch.nn.RNN.html.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        num_layers: Number of layers (RNNCell or GRUCell).
        dropout: Dropout rate.
        model_type: Either 'RNN' or 'GRU', to select which model we want.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            model_type: str,
        ):
        super().__init__()

        self.hidden_size = hidden_size
        model_class = RNNCell if model_type == 'RNN' else GRUCell

        self.layers = nn.ModuleList()
        self.layers.append(model_class(input_size, hidden_size, dropout))
        for i in range(1, num_layers):
          self.layers.append(model_class(hidden_size, hidden_size, dropout))

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor=None) -> tuple:
        """Pass the input sequence through all the RNN cells.
        Returns the output and the final hidden state of each RNN layer

        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Hidden state for each RNN layer.
                Can be None, in which case an initial hidden state is created.
                Shape of [batch_size, n_layers, hidden_size].

        Output
        ------
            y: Output embeddings for each token after the RNN layers.
                Shape of [batch_size, seq_len, hidden_size].
            h: Final hidden state.
                Shape of [batch_size, n_layers, hidden_size].
        """
        input = x
        h = torch.zeros([x.shape[0], len(self.layers), self.hidden_size], device=x.device) if h is None else h
        final_h = torch.zeros_like(h, device=x.device)
        for l in range(len(self.layers)):
          input, h_out = self.layers[l](input, h[:, l, :])
          final_h[:, l, :] = h_out

        return input, final_h


class GRU(nn.Module):
    """Implementation of a GRU based on https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        num_layers: Number of layers.
        dropout: Dropout rate.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ):
        super().__init__()

        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.layers.append(GRUCell(input_size, hidden_size, dropout))
        for i in range(1, num_layers):
          self.layers.append(GRUCell(hidden_size, hidden_size, dropout))

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor=None) -> tuple:
        """
        Args
        ----
            x: Input sequence
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state for each layer.
                If 'None', then an initial hidden state (a zero filled tensor)
                is created.
                Shape of [batch_size, n_layers, hidden_size].

        Output
        ------
            output:
                Shape of [batch_size, seq_len, hidden_size].
            h_n: Final hidden state.
                Shape of [batch_size, n_layers, hidden size].
        """
        input = x
        h = torch.zeros([x.shape[0], len(self.layers), self.hidden_size], device=x.device) if h is None else h
        final_h = torch.zeros_like(h, device=x.device)
        for l in range(len(self.layers)):
          input, h_out = self.layers[l](input, h[:, l, :])
          final_h[:, l, :] = h_out

        return input, final_h


class GRUCell(nn.Module):
    """A single GRU layer.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        dropout: Dropout rate.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            dropout: float,
        ):
        super().__init__()
        self.hidden_size = hidden_size

        self.Wiz = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.Whz = nn.Linear(hidden_size, hidden_size, device=DEVICE)
        self.Wir = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.Whr = nn.Linear(hidden_size, hidden_size, device=DEVICE)
        self.Win = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.Whn = nn.Linear(hidden_size, hidden_size, device=DEVICE)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor) -> tuple:
        """
        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state.
                Shape of [batch_size, hidden_size].

        Output
        ------
            y: Token embeddings.
                Shape of [batch_size, seq_len, hidden_size].
            h: Last hidden state.
                Shape of [batch_size, hidden_size].
        """
        batch_size, seq_len, input_size = x.shape
        y = torch.zeros([batch_size, seq_len, self.hidden_size], device=DEVICE)

        for t in range(seq_len):
          input = x[:, t, :]
          w_ir = self.Wir(input)
          w_hr = self.Whr(h)
          w_iz = self.Wiz(input)
          w_hz = self.Whz(h)
          w_in = self.Win(input)
          w_hn = self.Whn(h)
          r = torch.sigmoid(w_ir + w_hr)
          z = torch.sigmoid(w_iz + w_hz)
          n = torch.tanh(w_in + r * w_hn)
          h = z * h + (1 - z) * n
          y[:, t, :] = self.dropout(h)

        return y, h


class TranslationRNN(nn.Module):
    """Basic RNN encoder and decoder for a translation task.
    It can run as a vanilla RNN or a GRU-RNN.

    Parameters
    ----------
        n_tokens_src: Number of tokens in the source vocabulary.
        n_tokens_tgt: Number of tokens in the target vocabulary.
        dim_embedding: Dimension size of the word embeddings (for both language).
        dim_hidden: Dimension size of the hidden layers in the RNNs
            (for both the encoder and the decoder).
        n_layers: Number of layers in the RNNs.
        dropout: Dropout rate.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.
        model_type: Either 'RNN' or 'GRU', to select which model we want.
    """

    def __init__(
            self,
            n_tokens_src: int,
            n_tokens_tgt: int,
            dim_embedding: int,
            dim_hidden: int,
            n_layers: int,
            dropout: float,
            src_pad_idx: int,
            tgt_pad_idx: int,
            model_type: str,
        ):
        super().__init__()
        self.src_embeddings = nn.Embedding(n_tokens_src, dim_embedding, src_pad_idx)
        self.tgt_embeddings = nn.Embedding(n_tokens_tgt, dim_embedding, tgt_pad_idx)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.encoder = RNN(dim_embedding, dim_hidden, n_layers, dropout, model_type)
        self.norm = nn.LayerNorm(dim_hidden)
        self.decoder = RNN(dim_embedding, dim_hidden, n_layers, dropout, model_type)
        self.out_layer = nn.Linear(dim_hidden, n_tokens_tgt)


    def forward(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor
    ) -> torch.FloatTensor:
        """Predict the target tokens logits based on the source tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, src_seq_len].
            target: Batch of target sentences.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y: Distributions over the next token for all tokens in each sentences.
                Those need to be the logits only, do not apply a softmax because
                it will be done in the loss computation for numerical stability.
                See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for more informations.
                Shape of [batch_size, tgt_seq_len, n_tokens_tgt].
        """
        source = torch.fliplr(source)

        src_emb = self.src_embeddings(source)
        out, hidden = self.encoder(src_emb)

        hidden = self.norm(hidden)

        tgt_emb = self.tgt_embeddings(target)
        y, hidden = self.decoder(tgt_emb, hidden)

        y = self.out_layer(y)

        return y


###################
# Transformer Models
###################

class PositionalEncoding(nn.Module):
    """
    This PE module comes from:
    Pytorch. (2021). LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).to(DEVICE)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(DEVICE)
        pe = torch.zeros(max_len, 1, d_model).to(DEVICE)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = rearrange(x, "b s e -> s b e")
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        x = rearrange(x, "s b e -> b s e")
        return self.dropout(x)


def attention(
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.BoolTensor=None,
        dropout: nn.Dropout=None,
    ) -> tuple:
    """Computes multihead scaled dot-product attention from the
    projected queries, keys and values.

    Args
    ----
        q: Batch of queries.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        k: Batch of keys.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        v: Batch of values.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        mask: Prevent tokens to attend to some other tokens (for padding or autoregressive attention).
            Attention is prevented where the mask is `True`.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2],
            or broadcastable to that shape.
        dropout: Dropout layer to use.

    Output
    ------
        y: Multihead scaled dot-attention between the queries, keys and values.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        attn: Computed attention between the keys and the queries.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2].
    """
    d_k = q.size(-1)
    scores = torch.einsum('b h s d, b h t d -> b h s t', q, k) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.einsum('bhst,bhtd->bhsd', attn, v)
    return output, attn


class MultiheadAttention(nn.Module):
    """Multihead attention module.
    Can be used as a self-attention and cross-attention layer.
    The queries, keys and values are projected into multiple heads
    before computing the attention between those tensors.

    Parameters
    ----------
        dim: Dimension of the input tokens.
        n_heads: Number of heads. `dim` must be divisible by `n_heads`.
        dropout: Dropout rate.
    """
    def __init__(
            self,
            dim: int,
            n_heads: int,
            dropout: float,
        ):
        super().__init__()

        assert dim % n_heads == 0

        self.d_k = dim // n_heads
        self.n_heads = n_heads
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        
    def forward(
            self,
            q: torch.FloatTensor,
            k: torch.FloatTensor,
            v: torch.FloatTensor,
            key_padding_mask: torch.BoolTensor = None,
            attn_mask: torch.BoolTensor = None,
        ) -> torch.FloatTensor:
        """Computes the scaled multi-head attention form the input queries,
        keys and values.

        Project those queries, keys and values before feeding them
        to the `attention` function.

        The masks are boolean masks. Tokens are prevented to attends to
        positions where the mask is `True`.

        Args
        ----
            q: Batch of queries.
                Shape of [batch_size, seq_len_1, dim_model].
            k: Batch of keys.
                Shape of [batch_size, seq_len_2, dim_model].
            v: Batch of values.
                Shape of [batch_size, seq_len_2, dim_model].
            key_padding_mask: Prevent attending to padding tokens.
                Shape of [batch_size, seq_len_2].
            attn_mask: Prevent attending to subsequent tokens.
                Shape of [seq_len_1, seq_len_2].

        Output
        ------
            y: Computed multihead attention.
                Shape of [batch_size, seq_len_1, dim_model].
        """
        mask = None
        if key_padding_mask is not None:
            padding_mask = rearrange(key_padding_mask, 'b s -> b 1 1 s')
            mask = padding_mask

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            mask = attn_mask if mask is None else mask | attn_mask

        query, key, value = [
            rearrange(l(x), 'b s (h d) -> b h s d', h=self.n_heads)
            for l, x in zip(self.linears[:3], (q, k, v))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = rearrange(x, 'b h s d -> b s (h d)')

        return self.linears[-1](x)


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer.

    Parameters
    ----------
        d_model: The dimension of decoders inputs/outputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            nhead: int,
            dropout: float
        ):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            src: torch.FloatTensor,
            tgt: torch.FloatTensor,
            tgt_mask_attn: torch.BoolTensor,
            src_key_padding_mask: torch.BoolTensor,
            tgt_key_padding_mask: torch.BoolTensor,
        ) -> torch.FloatTensor:
        """Decode the next target tokens based on the previous tokens.

        Args
        ----
            src: Batch of source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of target sentences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y:  Batch of sequence of embeddings representing the predicted target tokens
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask_attn,
                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(tgt, src, src,
                           key_padding_mask=src_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    """Implementation of the transformer decoder stack.

    Parameters
    ----------
        d_model: The dimension of decoders inputs/outputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        num_decoder_layers: Number of stacked decoders.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_decoder_layer:int,
            nhead: int,
            dropout: float
        ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
              TransformerDecoderLayer(
              d_model=d_model, nhead=nhead, d_ff=d_ff,
              dropout=dropout)
              for _ in range(num_decoder_layer)
            ]
            )


    def forward(
            self,
            src: torch.FloatTensor,
            tgt: torch.FloatTensor,
            tgt_mask_attn: torch.BoolTensor,
            src_key_padding_mask: torch.BoolTensor,
            tgt_key_padding_mask: torch.BoolTensor,
        ) -> torch.FloatTensor:
        """Decodes the source sequence by sequentially passing.
        the encoded source sequence and the target sequence through the decoder stack.

        Args
        ----
            src: Batch of encoded source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of taget sentences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y:  Batch of sequence of embeddings representing the predicted target tokens
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        for layer in self.layers:
            tgt = layer(src, tgt, tgt_mask_attn, src_key_padding_mask, tgt_key_padding_mask)
        return tgt


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer.

    Parameters
    ----------
        d_model: The dimension of input tokens.
        dim_feedforward: Hidden dimension of the feedforward networks.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            nhead: int,
            dropout: float,
        ):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        src: torch.FloatTensor,
        key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Encodes the input. Does not attend to masked inputs.

        Args
        ----
            src: Batch of embedded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        src2 = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    """Implementation of the transformer encoder stack.

    Parameters
    ----------
        d_model: The dimension of encoders inputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        num_encoder_layers: Number of stacked encoders.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            dim_feedforward: int,
            num_encoder_layers: int,
            nheads: int,
            dropout: float
        ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
              TransformerEncoderLayer(
              d_model=d_model, nhead=nheads, d_ff=dim_feedforward,
              dropout=dropout)
              for _ in range(num_encoder_layers)
            ]
            )


    def forward(
            self,
            src: torch.FloatTensor,
            key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Encodes the source sequence by sequentially passing.
        the source sequence through the encoder stack.

        Args
        ----
            src: Batch of embedded source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source sequence.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        for layer in self.layers:
            src = layer(src, key_padding_mask)
        return src


class Transformer(nn.Module):
    """Implementation of a Transformer based on the paper: https://arxiv.org/pdf/1706.03762.pdf.

    Parameters
    ----------
        d_model: The dimension of encoders/decoders inputs/ouputs.
        nhead: Number of heads for each multi-head attention.
        num_encoder_layers: Number of stacked encoders.
        num_decoder_layers: Number of stacked encoders.
        dim_feedforward: Hidden dimension of the feedforward networks.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
        ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
            )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
            )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
            )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers
            )

    def forward(
            self,
            src: torch.FloatTensor,
            tgt: torch.FloatTensor,
            tgt_mask_attn: torch.BoolTensor,
            src_key_padding_mask: torch.BoolTensor,
            tgt_key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Compute next token embeddings.

        Args
        ----
            src: Batch of source sequences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of target sequences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y: Next token embeddings, given the previous target tokens and the source tokens.
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        enc = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        dec = self.decoder(
            tgt, enc, tgt_mask=tgt_mask_attn, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
            )
        return dec


class TranslationTransformer(nn.Module):
    """Basic Transformer encoder and decoder for a translation task.
    Manage the masks creation, and the token embeddings.
    Position embeddings can be learnt with a standard `nn.Embedding` layer.

    Parameters
    ----------
        n_tokens_src: Number of tokens in the source vocabulary.
        n_tokens_tgt: Number of tokens in the target vocabulary.
        n_heads: Number of heads for each multi-head attention.
        dim_embedding: Dimension size of the word embeddings (for both language).
        dim_hidden: Dimension size of the feedforward layers
            (for both the encoder and the decoder).
        n_layers: Number of layers in the encoder and decoder.
        dropout: Dropout rate.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.
    """
    def __init__(
            self,
            n_tokens_src: int,
            n_tokens_tgt: int,
            n_heads: int,
            dim_embedding: int,
            dim_hidden: int,
            n_layers: int,
            dropout: float,
            src_pad_idx: int,
            tgt_pad_idx: int,
        ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=dim_embedding,nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=dim_hidden, dropout=dropout, batch_first=True
            )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(n_tokens_src, dim_embedding)
        self.tgt_embedding = nn.Embedding(n_tokens_tgt, dim_embedding)

        self.positional_encoding = PositionalEncoding(dim_embedding, dropout)

        self.linear_out = nn.Linear(dim_embedding, n_tokens_tgt)

    def forward(
            self,
            source: torch.LongTensor,
            target: torch.LongTensor
        ) -> torch.FloatTensor:
        """Predict the target tokens logites based on the source tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, seq_len_src].
            target: Batch of target sentences.
                Shape of [batch_size, seq_len_tgt].

        Output
        ------
            y: Distributions over the next token for all tokens in each sentences.
                Those need to be the logits only, do not apply a softmax because
                it will be done in the loss computation for numerical stability.
                See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for more informations.
                Shape of [batch_size, seq_len_tgt, n_tokens_tgt].
        """
        src_embedded = self.src_embedding(source)
        src_embedded = self.positional_encoding(src_embedded)

        tgt_embedded = self.tgt_embedding(target)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        tgt_mask_attn = self.generate_causal_mask(target)
        src_key_padding_mask, tgt_key_padding_mask = self.generate_key_padding_mask(source, target)

        transformer_output = self.transformer(
            src=src_embedded, tgt=tgt_embedded, tgt_mask=tgt_mask_attn, src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.linear_out(transformer_output)
        return output

    def generate_causal_mask(
            self,
            target: torch.LongTensor,
        ) -> tuple:
        """Generate the masks to prevent attending subsequent tokens.

        Args
        ----
            target: Batch of target sentences.
                Shape of [batch_size, seq_len_tgt].

        Output
        ------
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [seq_len_tgt, seq_len_tgt].

        """
        seq_len = target.shape[1]

        tgt_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        tgt_mask = torch.triu(tgt_mask, diagonal=1).to(target.device)

        return tgt_mask

    def generate_key_padding_mask(
            self,
            source: torch.LongTensor,
            target: torch.LongTensor,
        ) -> tuple:
        """Generate the masks to prevent attending padding tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, seq_len_src].
            target: Batch of target sentences.
                Shape of [batch_size, seq_len_tgt].

        Output
        ------
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, seq_len_src].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, seq_len_tgt].

        """
        src_key_padding_mask = source == self.src_pad_idx
        tgt_key_padding_mask = target == self.tgt_pad_idx

        return src_key_padding_mask, tgt_key_padding_mask