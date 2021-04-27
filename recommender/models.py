import pytorch_lightning as pl
import torch
import torchvision.models as models
from torch.nn import Linear
from torch.nn import functional as F
import random

import torch.nn as nn


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


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
        decoder_layer = nn.TransformerDecoderLayer(
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
        batch_size, out_sequence_len = trg_items.size(0), trg_items.size(1)

        trg_items = self.item_embeddings(trg_items)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder)

        trg_items += pos_decoder

        trg = trg_items.permute(1, 0, 2)

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask)

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

        loss = F.l1_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src_items, src_features, trg_items, trg_out = batch

        y_hat = self((src_items, src_features, trg_items))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = F.l1_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src_items, src_features, trg_items, trg_out = batch

        y_hat = self((src_items, src_features, trg_items))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = F.l1_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":

    n_items = 1000

    recommender = Recommender(vocab_size=1000)

    src_items = torch.randint(low=0, high=n_items, size=(32, 30))
    src_features = torch.rand(32, 30, 1)

    trg_items = torch.randint(low=0, high=n_items, size=(32, 5))
    trg_out = torch.randint(low=0, high=n_items, size=(32, 5, 1))

    out = recommender((src_items, src_features, trg_items))

    print(out.shape)

    loss = recommender.training_step((src_items, src_features, trg_items, trg_out), batch_idx=1)

    print(loss)