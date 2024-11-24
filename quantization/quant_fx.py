import torch
import torch.quantization.quantize_fx as quantize_fx
from datetime import datetime
import argparse
from model.resnet2 import *
from catalyst import dl
from dataset import CifarDS
from config import *
import logging

# 디버깅용 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        default="{}/train.10.pth".format(WEIGHTS_DIR),
        help="Path to model to quantize",
    )
    return argparser


def calibrate(model):
    logging.info("Starting calibration...")
    dataset = CifarDS()
    train_loader = dataset.get_train_gen(batch_size=1)  # 배치 사이즈 1로 설정
    logging.info("Calibration dataset prepared. Number of batches: %d", len(train_loader))

    # 데이터 루프를 통해 캘리브레이션 수행
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            # 리스트/튜플에서 입력과 타겟 추출
            inputs = batch[0]  # 첫 번째 요소가 입력 데이터
            logging.info("Calibration batch %d/%d: Processing...", i + 1, len(train_loader))
            model(inputs)  # 모델에 입력 전달하여 캘리브레이션 수행
            if i >= 100:  # 캘리브레이션 배치 수 제한 (필요 시 수정)
                logging.info("Calibration limited to 100 batches.")
                break

    logging.info("Calibration completed.")


def main(args):
    logging.info("Loading model from path: %s", args.path)
    model = load_model(args.path)
    model.eval()

    logging.info("Preparing model for quantization...")
    qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
    
    # 예제 입력 생성
    example_inputs = torch.randn(1, 3, 32, 32)  # CIFAR-10 크기에 맞는 예제 입력
    logging.info("Example inputs created: %s", example_inputs.size())

    # 양자화 준비
    model = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs)
    logging.info("Quantization preparation completed.")

    # 캘리브레이션
    calibrate(model)

    # 양자화 변환
    logging.info("Converting model to quantized version...")
    model = quantize_fx.convert_fx(model)
    logging.info("Quantization conversion completed.")

    # 모델 저장
    model_name = "{}/qmodel_{}{}".format(
        QWEIGHTS_DIR, datetime.now().strftime("%H_%M_%S"), ".pth"
    )
    torch.jit.save(torch.jit.script(model), model_name)
    logging.info("Quantized model saved to: %s", model_name)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logging.info("Starting quantization script...")
    main(args)
    logging.info("Quantization script completed successfully.")
