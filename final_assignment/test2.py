# to test train1, modify without the train process
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 构建ResNet-18模型（已经通过自监督学习进行预训练）
backbone = models.resnet18(pretrained=False)
num_ftrs = backbone.fc.in_features
backbone.fc = nn.Identity()  # 移除原始的全连接层

# 构建线性分类器
classifier = nn.Linear(num_ftrs, 10)  # CIFAR-10有10个类别

# 加载自监督学习训练得到的模型参数
backbone.load_state_dict(torch.load('resnet18_selfsupervised.pth'))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

# 线性分类训练
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)
classifier.to(device)

'''for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        features = backbone(images)
        outputs = classifier(features)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
'''
# 在测试集上进行评估
backbone.eval()
classifier.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        features = backbone(images)
        outputs = classifier(features)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")
