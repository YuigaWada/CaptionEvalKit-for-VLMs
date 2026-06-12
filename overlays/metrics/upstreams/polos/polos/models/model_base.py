# -*- coding: utf-8 -*-
r"""
Model Base
==============
    Abstract base class used to build new modules inside Polos.
    This class is just an extention of PyTorch Lightning main module:
    https://pytorch-lightning.readthedocs.io/en/0.8.4/lightning-module.html
"""
from argparse import Namespace
from os import path
import os
from typing import Dict, Generator, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset, Dataset
from PIL import Image

import pytorch_lightning as ptl
from polos.models.encoders import Encoder, str2encoder
from polos.schedulers import str2scheduler

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ModelBase(ptl.LightningModule):
    """
    Extends PyTorch Lightning with a common structure and interface
    that will be shared across all architectures.

    :param hparams: Namespace with hyper-parameters
    """

    class ModelConfig:
        model: str = None
        encoder_learning_rate: float = 1e-06
        layerwise_decay: float = 1.0
        layer: str = "mix"
        scalar_mix_dropout: float = 0.0
        loss: str = "mse"
        hidden_sizes: str = "1024"
        activations: str = "Tanh"
        dropout: float = 0.1
        final_activation: str = "Sigmoid"
        batch_size: int = 8
        nr_frozen_epochs: int = 0
        keep_embeddings_frozen: bool = False
        optimizer: str = "Adam"
        learning_rate: float = 1e-05
        scheduler: str = "constant"
        warmup_steps: int = None
        encoder_model: str = "XLMR"
        pretrained_model: str = "xlmr.base"
        pool: str = "avg"
        load_weights: str = False
        train_path: str = None
        val_path: str = None
        test_path: str = None
        train_img_dir_path: str = None
        val_img_dir_path: str = None
        test_img_dir_path: str = None
        loader_workers: int = 8
        monitor: str = "kendall"

        def __init__(self, initial_data: dict) -> None:
            for key in initial_data:
                if hasattr(self, key):
                    setattr(self, key, initial_data[key])

        def namespace(self) -> Namespace:
            return Namespace(
                **{
                    name: getattr(self, name)
                    for name in dir(self)
                    if not callable(getattr(self, name)) and not name.startswith("__")
                }
            )

    def __init__(self, hparams: Namespace) -> None:
        super(ModelBase, self).__init__()
        self._hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.encoder = self._build_encoder()

        self._build_model()
        self._build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        if (
            hasattr(self.hparams, "keep_embeddings_frozen")
            and self.hparams.keep_embeddings_frozen
        ):
            self.encoder.freeze_embeddings()

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def _build_loss(self):
        pass

    def _build_model(self) -> ptl.LightningModule:
        if (
            hasattr(self.hparams, "load_weights")
            and self.hparams.load_weights
            and path.exists(self.hparams.load_weights)
        ):
            click.secho(f"Loading weights from {self.hparams.load_weights}", fg="red")
            self.load_weights(self.hparams.load_weights)

    def _build_encoder(self) -> Encoder:
        try:
            return str2encoder[self.hparams.encoder_model].from_pretrained(self.hparams)
        except KeyError:
            raise Exception(f"{self.hparams.encoder_model} invalid encoder model!")

    def _build_optimizer(self, parameters: Generator) -> torch.optim.Optimizer:
        if hasattr(torch.optim, self.hparams.optimizer):
            return getattr(torch.optim, self.hparams.optimizer)(
                params=parameters, lr=self.hparams.learning_rate
            )
        else:
            raise Exception(f"{self.hparams.optimizer} invalid optimizer!")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LambdaLR:
        return str2scheduler[self.hparams.scheduler].from_hparams(
            optimizer=optimizer, hparams=self.hparams
        )

    def freeze_encoder(self) -> None:
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze()

    def get_sentence_embedding(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        pooling: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]]:
        encoder_out = self.encoder(tokens, lengths)
        wordemb = encoder_out["wordemb"]
        sentemb = encoder_out["sentemb"]
        all_layers = encoder_out["all_layers"]
        mask = encoder_out["mask"]
        padding_index = self.encoder.padding_index

        if self.hparams.layer == "mix":
            if self.scalar_mix:
                wordemb = self.scalar_mix(all_layers, mask)
            else:
                wordemb = torch.mean(torch.stack(all_layers), dim=0)
        else:
            wordemb = all_layers[self.layer]

        if not pooling:
            return sentemb, wordemb, mask, padding_index

        return self.pool_sentence_embedding(
            sentemb=sentemb,
            wordemb=wordemb,
            mask=mask,
            pooling=self.hparams.pool,
            padding_index=padding_index,
        )

    def load_weights(self, checkpoint: str) -> None:
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint["state_dict"], strict=False)

    def pool_sentence_embedding(
        self,
        sentemb: torch.Tensor,
        wordemb: torch.Tensor,
        mask: torch.Tensor,
        pooling: str,
        padding_index: int,
    ) -> torch.Tensor:
        if pooling == "default":
            return sentemb
        if pooling == "max":
            sentemb = max_pooling(wordemb, mask, padding_index)
        elif pooling == "avg":
            sentemb = average_pooling(wordemb, mask)
        elif pooling == "cls":
            sentemb = wordemb[:, 0, :]
        elif pooling == "cls+avg":
            cls_sentemb = wordemb[:, 0, :]
            avg_sentemb = average_pooling(wordemb, mask)
            sentemb = torch.cat((cls_sentemb, avg_sentemb), dim=1)
        else:
            raise Exception("Invalid pooling technique.")
        return sentemb

    def predict(
        self,
        samples: List[Dict[str, str]],
        cuda: bool = False,
        show_progress: bool = True,
        batch_size: int = -1,
    ) -> (Dict[str, Union[str, float]], List[float]):
        if self.training:
            self.eval()

        if cuda and torch.cuda.is_available():
            self.to("cuda")

        batch_size = self.hparams.batch_size if batch_size < 1 else batch_size
        with torch.no_grad():
            batches = [
                samples[i : i + batch_size] for i in range(0, len(samples), batch_size)
            ]
            model_inputs = []
            if show_progress:
                from tqdm import tqdm
                pbar = tqdm(total=len(batches), desc="Preparing batches...", dynamic_ncols=True, leave=None)
            for batch in batches:
                batch = self.prepare_sample(batch, inference=True)
                model_inputs.append(batch)
                if show_progress:
                    pbar.update(1)
            if show_progress:
                pbar.close()

            if show_progress:
                from tqdm import tqdm
                pbar = tqdm(total=len(batches), desc="Scoring hypothesis...", dynamic_ncols=True, leave=None)
            scores = []
            for model_input in model_inputs:
                if cuda and torch.cuda.is_available():
                    model_input = move_to_cuda(model_input)
                    model_out = self.forward(**model_input)
                    model_out = move_to_cpu(model_out)
                else:
                    model_out = self.forward(**model_input)

                model_scores = model_out["score"].numpy().tolist()
                for i in range(len(model_scores)):
                    scores.append(model_scores[i][0])
                if show_progress:
                    pbar.update(1)
            if show_progress:
                pbar.close()

        assert len(scores) == len(samples)
        for i in range(len(scores)):
            samples[i]["predicted_score"] = scores[i]
        return samples, scores


def move_to_cuda(sample):
    from polos.models.utils import move_to_cuda as _move_to_cuda
    return _move_to_cuda(sample)


def move_to_cpu(sample):
    from polos.models.utils import move_to_cpu as _move_to_cpu
    return _move_to_cpu(sample)


def average_pooling(*args, **kwargs):
    from polos.models.utils import average_pooling as _average_pooling
    return _average_pooling(*args, **kwargs)


def max_pooling(*args, **kwargs):
    from polos.models.utils import max_pooling as _max_pooling
    return _max_pooling(*args, **kwargs)
