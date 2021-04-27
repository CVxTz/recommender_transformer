from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch.nn import Module, MultiheadAttention, Dropout, LayerNorm
from torch.nn import functional as F


class CustomTransformerDecoderLayer(Module):
    """
    No self attention in the decoder
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(CustomTransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Recommender(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=128,
        n_features=1,
        dropout=0.4,
        lr=1e-4,
    ):
        super().__init__()

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.item_embeddings = torch.nn.Embedding(
            self.vocab_size, embedding_dim=channels
        )

        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout
        )
        decoder_layer = CustomTransformerDecoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.project = Linear(n_features + channels, channels)

        self.linear_out = Linear(channels, 1)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items, src_features):
        src_items = self.item_embeddings(src_items)

        src = torch.cat(tensors=[src_items, src_features], dim=-1)

        src = self.project(src)

        batch_size, in_sequence_len = src.size(0), src.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src += pos_encoder

        src = src.permute(1, 0, 2)

        src = self.encoder(src)

        return src

    def decode_trg(self, trg_items, memory):
        trg_items = self.item_embeddings(trg_items)

        trg = trg_items.permute(1, 0, 2)

        out = self.decoder(tgt=trg, memory=memory)

        out = out.permute(1, 0, 2)

        out = self.linear_out(out)

        return out

    def forward(self, x):
        src_items, src_features, trg_items = x

        src = self.encode_src(src_items, src_features)

        out = self.decode_trg(trg_items=trg_items, memory=src)

        return out

    def training_step(self, batch, batch_idx):
        src_items, src_features, trg_items, trg_out = batch

        y_hat = self((src_items, src_features, trg_items))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src_items, src_features, trg_items, trg_out = batch

        y_hat = self((src_items, src_features, trg_items))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = F.mse_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src_items, src_features, trg_items, trg_out = batch

        y_hat = self((src_items, src_features, trg_items))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = F.mse_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
