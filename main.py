import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from torchvision import models

from dataloader import IncrementalMNIST


# iCaRL 모델 클래스 정의
class iCaRLModel(nn.Module):
    def __init__(self, feature_extractor, num_classes, memory_size=2000, device = "cuda"):
        super(iCaRLModel, self).__init__()
        self.feature_extractor = feature_extractor
        
        # Classification layer 초기화
        self.classifier = nn.Linear(self.feature_extractor.fc.in_features, num_classes)
        self.feature_extractor.fc = nn.Identity()  # Feature extractor에서 fully connected layer 제거

        # 메모리 관련 초기화
        self.exemplar_sets = []  # 각 클래스의 프로토타입
        self.memory_size = memory_size  # 메모리 제한 (총 메모리 크기)

        # 학습된 클래스 개수
        self.num_classes = 0
        
        # device type
        self.device = torch.device(device)

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        return outputs

    def extract_features(self, x):
        # Feature 추출만 수행
        return self.feature_extractor(x)

    def classify(self, x):
        # 프로토타입 기반으로 유사도 계산 후 분류
        extracted_features = self.extract_features(x).detach().cpu().numpy()
        predictions = []

        # 현재까지 각 클래스의 프로토타입과 유사도 비교
        for features in extracted_features:
            similarities = [
                np.linalg.norm(features - exemplar.mean(axis=0))
                for exemplar in self.exemplar_sets
            ]
            predictions.append(np.argmin(similarities))

        return torch.tensor(predictions)

    def update_representation(self, train_loader, num_epochs=10, learning_rate=0.01):
        # 모델 학습 (새로 추가된 클래스 데이터 사용)
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        self.to(self.device)

        for epoch in range(num_epochs):
            self.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Knowledge Distillation 손실 계산
                if self.num_classes > 0:
                    old_outputs = self.forward(images)[:, :self.num_classes].detach()
                    outputs = self.forward(images)
                    loss = self.knowledge_distillation_loss(outputs, old_outputs, labels)
                else:
                    outputs = self.forward(images)
                    loss = criterion(outputs, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 학습된 클래스 수 업데이트
        self.num_classes += len(set(labels.tolist()))

    def knowledge_distillation_loss(self, outputs, old_outputs, labels, alpha=0.5, temperature=2):
        # Knowledge Distillation 손실 함수
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.MSELoss()

        # Cross Entropy
        loss_ce = criterion_ce(outputs, labels)

        # Knowledge Distillation 손실
        if self.num_classes > 0:
            outputs = F.softmax(outputs[:, :self.num_classes] / temperature, dim=1)
            old_outputs = F.softmax(old_outputs / temperature, dim=1)
            loss_kd = criterion_kd(outputs, old_outputs) * (temperature ** 2)
        else:
            loss_kd = 0

        # 손실 합산
        return alpha * loss_ce + (1 - alpha) * loss_kd

    def update_memory(self, train_loader, num_new_classes):
        # 메모리에 새로운 프로토타입 추가 (Exemplar Set 갱신)
        num_exemplars_per_class = self.memory_size // (self.num_classes + num_new_classes)

        # 새롭게 추가된 클래스에 대해 프로토타입 생성
        for class_idx in range(num_new_classes):
            class_data = []
            for images, labels in train_loader:
                for i in range(len(labels)):
                    if labels[i] == class_idx:
                        class_data.append(images[i].numpy())

            class_data = torch.tensor(class_data)
            features = self.extract_features(class_data).detach().cpu().numpy()

            # K-means와 같은 방식으로 메모리에서 저장할 데이터 선정 (평균과 가장 가까운 데이터 선택)
            exemplar = []
            class_mean = np.mean(features, axis=0)
            selected_indices = set()

            for _ in range(num_exemplars_per_class):
                distances = [
                    np.linalg.norm(feature - class_mean) for idx, feature in enumerate(features) if idx not in selected_indices
                ]
                chosen_idx = np.argmin(distances)
                exemplar.append(features[chosen_idx])
                selected_indices.add(chosen_idx)

            self.exemplar_sets.append(np.array(exemplar))

from tqdm import tqdm

# 사용 예시
if __name__ == "__main__":
    # ResNet18을 feature extractor로 사용
    feature_extractor = models.resnet18(pretrained=True)
    num_classes = 10  # 총 클래스 수
    model = iCaRLModel(feature_extractor, num_classes)

    # IncrementalMNIST 데이터셋 사용
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1채널 이미지를 3채널로 복사
                                    ])
    incremental_mnist = IncrementalMNIST(root='./data', train=True, transform=transform, num_increments=5, batch_size=256, increment_type='class')
    test_incremental_mnist = IncrementalMNIST(root='./data', train=False, transform=transform, num_increments=5, batch_size=256, increment_type='class')

    # 전체 증분 단계 학습 및 평가
    for increment in range(incremental_mnist.num_increments):
        print(f"\nTraining Increment {increment + 1}/{incremental_mnist.num_increments}")

        # 현재 증분의 데이터 로더 가져오기
        train_loader = incremental_mnist.get_incremental_loader(increment)

        # tqdm을 통해 학습 과정 모니터링
        for epoch in range(5):  # 각 증분 단계마다 5 epochs로 학습
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Increment {increment + 1}, Epoch {epoch + 1}")
                model.train()  # 학습 모드로 변경
                for images, labels in tepoch:
                    images, labels = images.to(model.device), labels.to(model.device)
                    
                    # 모델 학습 업데이트
                    model.update_representation(train_loader, num_epochs=1, learning_rate=0.01)
                    tepoch.set_postfix({"increment": increment + 1, "epoch": epoch + 1})

        # 메모리 업데이트 (각 증분마다 새로운 클래스를 추가)
        print(f"Updating memory for increment {increment + 1}")
        model.update_memory(train_loader, num_new_classes=2)  # 예시로 각 증분마다 2개의 새 클래스 추가

        # 테스트 데이터셋 평가
        print(f"\nEvaluating Increment {increment + 1}/{incremental_mnist.num_increments} on test set")
        test_loader = test_incremental_mnist.get_incremental_loader(increment)
        model.eval()  # 평가 모드로 변경

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, unit="batch", desc=f"Testing Increment {increment + 1}"):
                images, labels = images.to(model.device), labels.to(model.device)
                predictions = model.classify(images)
                correct += (predictions == labels.cpu()).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Accuracy for Increment {increment + 1}: {accuracy:.4f}")

    # 학습 종료 후 최종 분류 예시
    print("\nClassification Results after Full Training:")
    for increment in range(incremental_mnist.num_increments):
        print(f"\nClassifying Increment {increment + 1}/{incremental_mnist.num_increments}")
        train_loader = incremental_mnist.get_incremental_loader(increment)

        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images = images.to(model.device)
                predictions = model.classify(images)
                print("Predictions:", predictions)
                print("Labels:", labels)
                break