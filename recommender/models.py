from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F


def masked_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):

    _, predicted = torch.max(y_pred, 1)

    y_true = torch.masked_select(y_true, mask)
    predicted = torch.masked_select(predicted, mask)

    acc = (y_true == predicted).double().mean()

    return acc


def masked_ce(y_pred, y_true, mask):

    loss = F.cross_entropy(y_pred, y_true, reduction="none")

    loss = loss * mask

    return loss.sum() / (mask.sum() + 1e-8)


class Recommender(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=128,
        cap=0,
        mask=1,
        dropout=0.4,
        lr=1e-4,
    ):
        super().__init__()

        self.cap = cap
        self.mask = mask

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.item_embeddings = torch.nn.Embedding(
            self.vocab_size, embedding_dim=channels
        )

        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.linear_out = Linear(channels, self.vocab_size)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items):
        src_items = self.item_embeddings(src_items)

        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src_items += pos_encoder

        src = src_items.permute(1, 0, 2)

        src = self.encoder(src)

        return src.permute(1, 0, 2)

    def forward(self, src_items):

        src = self.encode_src(src_items)

        out = self.linear_out(src)

        return out

    def training_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }
