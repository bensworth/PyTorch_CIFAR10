import unittest
import os
import pytest
import torch
import torchvision.models
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from fnn.cifar_ben.data import CIFAR10Data
from fnn.cifar_ben.module import CIFAR10Module
from fnn.cifar_ben.flexible_module_checkpoint import FlexibleModelCheckpoint


class TestModel:
    def test_model(self):
        args = Args()
        args.data_dir = "/tmp" #os.environ["FNN_DATA"]
        args.gpu_id = "0"
        args.logger = "tensorboard"
        args.classifier = "vgg11_bn"
        args.test_phase = 0
        args.dev = 0
        args.pretrained = 0
        args.precision = 32
        args.batch_size = 256
        args.max_epochs = 1
        args.num_workers = 0
        args.learning_rate = 1e-2
        args.weight_decay = 1e-2
        args.shuffle = False

        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="cifar10")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("cifar10", name=args.classifier)

        seed_everything(0)
        model = CIFAR10Module(args.__dict__)
        data = CIFAR10Data(args)
        n = len(data.train_dataloader())
        checkpoint = FlexibleModelCheckpoint(monitor="acc/val", mode="max", save_last=False,
                                             filename="{epoch}-{step}-{val_loss:.2f}",
                                             train_step_region=[(0, 1), (3, 4)],
                                             every_n_val_epochs=1)
        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else None,
            deterministic=True,
            weights_summary=None,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            callbacks=checkpoint,
            precision=args.precision,
            # Only run a small number of batches in this unit test.
            limit_train_batches=0.03, limit_val_batches=0.03, limit_test_batches=0.03,
        )

        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, data.test_dataloader())
        else:
            trainer.fit(model, data)
            test_result = trainer.test()
            assert len(test_result) == 1
            accuracy = test_result[0]["acc/test"]
            # We can't seem to get deterministic results, so use a range for now.
            assert 7 <= accuracy <= 14

class Args: pass
