import torch
import argparse
from catalyst import dl
from dataset import CifarDS
from glob import glob
import os
from config import *


def build_parser():
    parser = argparse.ArgumentParser()
    parser.prog = "validate model on CIFAR10"
    latest_model = max(glob("{}/*".format(QWEIGHTS_DIR)), key=os.path.getctime)
    parser.add_argument("--path", default=latest_model, type=str, help="path to model")
    parser.add_argument("-n", default=40, type=int, help="number of batches to valid")
    return parser

def validate_model(path, batch_size=1, num_batches=40):
    # 모델 로드
    model = torch.jit.load(path)
    model.eval()

    # 데이터 로드
    dataset = CifarDS()
    valid_loader = dataset.get_valid_gen(batch_size=batch_size)

    # 평가
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            if i >= num_batches:
                break

            inputs = batch[0]  # Assuming inputs are at index 0
            targets = batch[1]  # Assuming targets are at index 1

            outputs = model(inputs)
            _, predicted = outputs.topk(3, dim=1, largest=True, sorted=True)

            correct_top1 += (predicted[:, 0] == targets).sum().item()
            correct_top3 += sum(targets[i] in predicted[i] for i in range(len(targets)))
            total += targets.size(0)

    print(f"Top-1 Accuracy: {correct_top1 / total * 100:.2f}%")
    print(f"Top-3 Accuracy: {correct_top3 / total * 100:.2f}%")


def main(args):
    # model = torch.jit.load(args.path)
    # print(model)
    # model.eval()  # 수정: evaluate -> eval

    # dataset = CifarDS()
    # valid_loader = dataset.get_valid_gen(args.n)

    # runner = dl.SupervisedRunner(
    #     input_key="img", output_key="logits", target_key="targets", loss_key="loss"
    # )

    # runner.evaluate_loader(
    #     model=model,
    #     loader=valid_loader,
    #     callbacks=[
    #         dl.AccuracyCallback(input_key="logits", target_key="targets",),
    #         dl.PrecisionRecallF1SupportCallback(
    #             input_key="logits", target_key="targets", num_classes=10,
    #         ),
    #     ],
    #     verbose=True,
    # )
    validate_model("/workspace/ESE-framework/qweights/qmodel_07_37_23.pth")



if __name__ == "__main__":
    argparser = build_parser()
    main(argparser.parse_args())