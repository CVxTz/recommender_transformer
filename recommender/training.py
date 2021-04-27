import random

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from recommender.models import Recommender
from recommender.data_processing import split_df, df_to_np, pad_list, map_column


def shuffle(l1, l2):
    l3 = list(zip(l1, l2))

    random.shuffle(l3)

    l1, l2 = zip(*l3)

    return l1, l2


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, groups, grp_by, split, items_max, history_size=60, horizon_size=5
    ):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.history_size = history_size
        self.horizon_size = horizon_size
        self.items_max = items_max

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        src, trg = split_df(df, split=self.split)

        src_items = src["movieId_mapped"].tolist()

        src_items = pad_list(src_items, history_size=self.history_size)
        src_features = df_to_np(src[["rating"]], expected_size=self.history_size)

        trg_items = trg["movieId_mapped"].tolist() + [random.randint(0, self.items_max)]
        trg_out = trg["rating"].tolist() + [1.0]

        src_items = torch.tensor(src_items, dtype=torch.long)
        src_features = torch.tensor(src_features, dtype=torch.float)

        trg_items = torch.tensor(trg_items, dtype=torch.long)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src_items, src_features, trg_items, trg_out


def train(
    data_csv_path: str,
    log_dir: str = "recommender_logs",
    model_dir: str = "recommender_models",
    batch_size: int = 32,
    epochs: int = 2000,
    history_size: int = 60,
    horizon_size: int = 5,
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
        horizon_size=horizon_size,
        items_max=max(mapping.values()),
    )
    val_data = Dataset(
        groups=groups,
        grp_by=grp_by_train,
        split="val",
        history_size=history_size,
        horizon_size=horizon_size,
        items_max=max(mapping.values()),
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
        vocab_size=len(mapping) + 1,
        n_features=1,
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
