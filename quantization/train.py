import os

import torch
from torch import nn
from catalyst import dl

from collections import OrderedDict

from model.resnet2 import resnet18
import argparse

from dataset import CifarDS
from config import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def build_parser():
    argparser = argparse.ArgumentParser()
    argparser.prog = "train resnet20 on CIFAR10"
    argparser.add_argument(
        "--epochs", default=10, type=int, help="number of epochs to train"
    )
    return argparser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    runner = dl.SupervisedRunner(
        input_key="img", output_key="logits", target_key="targets", loss_key="loss"
    )

    dataset = CifarDS()
    train_loader = dataset.get_train_gen()
    valid_loader = dataset.get_valid_gen()
    loaders = OrderedDict({"train": train_loader, "valid": valid_loader})

    model = resnet18()
    model.load_state_dict(torch.load('model_weights.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        loggers={"tensorboard": dl.TensorboardLogger(logdir=TENSORBOARD_DIR)},
        num_epochs=args.epochs,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets"),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10,
            ),
            dl.CheckpointCallback(
            logdir=WEIGHTS_DIR,
            save_n_best=1,  # 가장 성능이 좋은 가중치만 저장
            mode="model",   # 모델 가중치만 저장
        ),
        ],
        logdir=LOG_DIR,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())