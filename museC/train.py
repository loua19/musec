"""Contains training code for training using PyTorch Lightning."""

import re
import argparse
import torch
import lightning as pl

from typing import Optional
from torch import nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from model import TransformerLM, ModelConfig
from utils import Tokenizer, Dataset


class PretrainLM(pl.LightningModule):
    """PyTorch Lightning Module for TransformerLM."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerLM(model_config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.pad_id)

    def forward(self, src: torch.Tensor):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self.model(src).transpose(1, 2)
        loss = self.loss_fn(logits, tgt)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # Not sure what this does
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self.model(src).transpose(1, 2)
        loss = self.loss_fn(logits, tgt).item()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # Not sure what this does
        )

    def configure_optimizers(self):  # Change to LLAMA
        # From LLaMa
        lr = 3e-4
        betas = (0.9, 0.95)
        weight_decay = 0.1
        T_max = 10
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=0.1 * lr,
            verbose=True,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train(
    checkpoint: Optional[str],
    data: str,
    workers: int,
    gpus: int,
    epochs: int,
):
    batch_size = 32
    model_config = ModelConfig()
    tokenizer = Tokenizer(model_config)

    if isinstance(checkpoint, str) and checkpoint is not None:
        model = PretrainLM.load_from_checkpoint(checkpoint)
    elif checkpoint is None:
        model = PretrainLM(model_config)

    dataset_train = Dataset.from_json(data, tokenizer, key="train")
    dataset_val = Dataset.from_json(data, tokenizer, key="val")
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=workers
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, num_workers=workers
    )

    # See https://shorturl.at/AGHZ3
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{train_loss}-{val_loss}",
        save_last=True,
        save_top_k=5,
        monitor="val_loss",
        save_weights_only=False,
    )

    a100_re = re.compile(r"[aA]100")
    v100_re = re.compile(r"[vV]100")
    if a100_re.search(torch.cuda.get_device_name(0)):
        print("A100 detected")
        prec = "bf16"
    elif v100_re.search(torch.cuda.get_device_name(0)):
        print("V100 detected")
        prec = "16-mixed"
    else:
        print("GPU not A100 or V100")
        prec = "16-mixed"

    trainer = pl.Trainer(
        devices=gpus,
        accelerator="gpu",
        precision=prec,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # Train
    trainer.fit(model, dataloader_train, dataloader_val)


def get_torch_module(load_path: str):
    """Extracts the PyTorch module from a checkpointed Lightning module.

    Args:
        load_path (str): Load path for checkpointed Lightning module.

    Returns:
        nn.Module: Module extracted from self.module
    """
    lightning_module = PretrainLM.load_from_checkpoint(load_path)

    return lightning_module.model


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument("-c", "--checkpoint")
    argp.add_argument("-d", "--data", type=str)
    argp.add_argument("--workers", type=int, default=1)
    argp.add_argument("--gpus", type=int, default=1)
    argp.add_argument("--epochs", type=int, required=True)
    kwargs = vars(argp.parse_args())

    return kwargs


if __name__ == "__main__":
    kwargs = parse_arguments()
    train(**kwargs)
