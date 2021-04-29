import random

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from recommender.models import Recommender
from recommender.data_processing import get_context, pad_list, map_column, MASK


def mask_list(l1, p=0.8):

    l1 = [a if random.random() < p else MASK for a in l1]

    return l1


def mask_last_elements_list(l1, val_context_size: int = 5):

    l1 = l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.5)

    return l1


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, history_size=120):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.history_size = history_size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        context = get_context(df, split=self.split, context_size=self.history_size)

        trg_items = context["movieId_mapped"].tolist()

        if self.split == "train":
            src_items = mask_list(trg_items)
        else:
            src_items = mask_last_elements_list(trg_items)

        pad_mode = "left" if random.random() < 0.5 else "right"
        trg_items = pad_list(trg_items, history_size=self.history_size, mode=pad_mode)
        src_items = pad_list(src_items, history_size=self.history_size, mode=pad_mode)

        src_items = torch.tensor(src_items, dtype=torch.long)

        trg_items = torch.tensor(trg_items, dtype=torch.long)

        return src_items, trg_items


def train(
    data_csv_path: str,
    log_dir: str = "recommender_logs",
    model_dir: str = "recommender_models",
    batch_size: int = 32,
    epochs: int = 2000,
    history_size: int = 120,
):
    data = pd.read_csv(data_csv_path)

    data.sort_values(by="timestamp", inplace=True)

    data, mapping, inverse_mapping = map_column(data, col_name="movieId")

    grp_by_train = data.groupby(by="userId")

    groups = list(grp_by_train.groups)

    train_data = Dataset(
        groups=groups,
        grp_by=grp_by_train,
        split="train",
        history_size=history_size,
    )
    val_data = Dataset(
        groups=groups,
        grp_by=grp_by_train,
        split="val",
        history_size=history_size,
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )

    model = Recommender(
        vocab_size=len(mapping) + 2,
        lr=1e-4,
        dropout=0.3,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="recommender",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(test_dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    print(output_json)

    return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    train(
        data_csv_path=args.data_csv_path,
        epochs=args.epochs,
    )
