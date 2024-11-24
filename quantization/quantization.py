import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub
from tqdm import tqdm
import numpy as np
import random
import argparse

# 랜덤 시드 고정
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ResNet 블록 정의
class QuantizedBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(QuantizedBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)
        return out

# ResNet18 모델 정의
class QuantizedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizedResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(QuantizedBasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(QuantizedBasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# BatchNorm + Conv2D 합치기
def fuse_model(model):
    for module_name, module in model.named_children():
        if isinstance(module, QuantizedBasicBlock):
            torch.quantization.fuse_modules(
                module, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True
            )
        elif hasattr(module, "named_children"):
            fuse_model(module)

# float의 경우 weight를 처리    
def apply_simulated_quantization(model, dtype):
    """
    Simulates quantization by converting weights to the specified dtype.
    Args:
        model (nn.Module): The model to apply simulated quantization.
        dtype (torch.dtype): The target dtype (e.g., torch.float16, torch.float32).
    """
    print(f"[INFO] Simulating {dtype} quantization.")
    for name, param in model.named_parameters():
        param.data = param.data.to(dtype)
    for name, buffer in model.named_buffers():
        buffer.data = buffer.data.to(dtype)
    model.simulated_dtype = dtype  # 모델에 현재 데이터 타입 기록

# 학습 함수
def train_model(model, trainloader, criterion, optimizer, scheduler, num_epochs=1, device="cuda"):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        print()
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False, ncols=100)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{running_loss / (i + 1):.4f}"})
        scheduler.step()
    print("Training complete.")

# 평가 함수
def evaluate_model(model, testloader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    simulated_dtype = getattr(model, "simulated_dtype", None)  # 모델에 기록된 데이터 타입 확인

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if simulated_dtype:  # 부동소수점 양자화 모드인 경우
                inputs = inputs.to(simulated_dtype)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# 메인 실행 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Quantization bits (default: 8)")
    args = parser.parse_args()

    set_seed()

    # 데이터 증강 및 데이터셋 로드
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

    # 모델 생성
    model = QuantizedResNet18(num_classes=10)

    # 학습 준비
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    # 모델 학습
    train_model(model, trainloader, criterion, optimizer, scheduler, num_epochs=150, device=device)

    # 평가
    evaluate_model(model, testloader, device=device)

    # PTQ 적용 시작
    model.eval()
    model.to("cpu")

    # Fuse 처리
    print("\n[INFO] Fusing model...")
    fuse_model(model)

    if args.bits in [8, 4, 2]:
        print("\n[INFO] Preparing model for quantization...")
        model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(model, inplace=True)

        print("[INFO] Calibrating model with test data...")
        with torch.no_grad():
            for inputs, _ in tqdm(testloader, desc="Calibration", leave=False):
                inputs = inputs.to("cpu")
                model(inputs)

        print("[INFO] Converting model to quantized version...")
        torch.quantization.convert(model, inplace=True)
    elif args.bits == 16:
        apply_simulated_quantization(model, torch.float16)
    elif args.bits == 32:
        apply_simulated_quantization(model, torch.float32)

    # Quantized 모델 평가
    evaluate_model(model, testloader, device="cpu")