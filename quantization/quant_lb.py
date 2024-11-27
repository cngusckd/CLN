from qmodel import qresnet18
from model.resnet2 import load_model
import argparse
from catalyst import dl

# from config import *
from fusion import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_parser():
    argparser = argparse.ArgumentParser()
    argparser.prog = "validate model on CIFAR10"
    argparser.add_argument(
        "--path",
        default="../weights/train.10.pth",
        type=str,
        help="path to model",
    )
    argparser.add_argument(
        "--nbits",
        default=8,
        type=int,
        help="n bits to quantize",
    )
    return argparser


def fit(model: nn.Module, n_batches: int) -> None:
    from dataset import CifarDS

    dataset = CifarDS()
    train_loader = dataset.get_train_gen(n_batches)

    runner = dl.SupervisedRunner(
        input_key="img", output_key="logits", target_key="targets", loss_key="loss"
    )

    runner.evaluate_loader(
        model=model.to(device),
        loader=train_loader,  # 단일 데이터 로더로 전달
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets"),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
        ],
        verbose=True,
    )


def evaluate(model: nn.Module, n_batches: int) -> None:
    from dataset import CifarDS

    dataset = CifarDS()
    valid_loader = dataset.get_valid_gen(n_batches)

    runner = dl.SupervisedRunner(
        input_key="img", output_key="logits", target_key="targets", loss_key="loss"
    )

    runner.evaluate_loader(
        model=model.to(device),
        loader=valid_loader,  # 단일 데이터 로더로 전달
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets",),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
        ],
        verbose=True,
    )


def main(args):
    # init
    model = load_model(args.path).to(device).eval()
    qmodel = qresnet18(int(args.nbits)).to(device)
    # prepare
    fuse_conv_bn(model)
    prepare(model, qmodel)
    # calibrate
    fit(qmodel, 2)
    # quantize
    qmodel.quantize()
    # save dict
    # torch.save(qmodel.state_dict(), PATH)
    # evaluate
    evaluate(qmodel, 40)


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())