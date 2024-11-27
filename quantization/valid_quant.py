import torch
import argparse
from model.resnet2 import resnet18
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

import time

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
    total_latency = 0.0

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # if i >= num_batches:
            #     break

            inputs = batch[0]  # Assuming inputs are at index 0
            targets = batch[1]  # Assuming targets are at index 1

            # Measure latency
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            total_latency += (end_time - start_time)

            _, predicted = outputs.topk(3, dim=1, largest=True, sorted=True)

            correct_top1 += (predicted[:, 0] == targets).sum().item()
            correct_top3 += sum(targets[i] in predicted[i] for i in range(len(targets)))
            total += targets.size(0)
    print(i)
    print(len(valid_loader))

    avg_latency = total_latency / len(valid_loader)
    print(f"Top-1 Accuracy: {correct_top1 / total * 100:.2f}%")
    print(f"Top-3 Accuracy: {correct_top3 / total * 100:.2f}%")
    print(f"Average Latency per Inference: {avg_latency * 1000:.2f} ms")



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
    validate_model("/workspace/CLN/qweights/qmodel_12_35_42.pth")
    validate_original_model("model_weights.pth")

def validate_original_model(path):
    # 모델 초기화 및 가중치 로드
    model = resnet18()
    model.load_state_dict(torch.load(path))
    model.eval()
    # model.cuda()  # GPU 사용 시 주석 해제

    # 더미 입력 데이터 생성 (예: 배치 크기 1, 채널 3, 224x224 이미지)
    dummy_input = torch.randn(1, 3, 32, 32)
    # dummy_input = dummy_input.cuda()  # GPU 사용 시 주석 해제

    # 워밍업 (GPU 초기화 및 캐시 로드)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 추론 시간 측정
    num_iterations = 10000
    total_time = 0.0

    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            total_time += (end_time - start_time)

    # 평균 추론 시간 계산 (밀리초 단위)
    avg_inference_time_ms = (total_time / num_iterations) * 1000

    print(f"모델 평균 추론 시간: {avg_inference_time_ms:.6f} ms")
    


if __name__ == "__main__":
    argparser = build_parser()
    main(argparser.parse_args())