# a rotate train
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 数据预处理
transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 构建ResNet-18模型
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 自监督任务有4个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 自监督训练
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, _ in train_dataloader:
        images = images.to(device)

        # 生成自监督任务标签（图像旋转）
        targets = torch.randint(low=0, high=4, size=(images.size(0),)).to(device)
        rotated_images = torch.transpose(images, 2, 3)  # 旋转图像
        # 前向传播
        outputs = model(rotated_images)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 保存旋转后的图像示例
    if (epoch + 1) % 5 == 0:
        save_image(rotated_images, f"rotated_images_epoch_{epoch+1}.png", normalize=True)

# 保存训练好的模型
torch.save(model.state_dict(), 'resnet18_rotate_selfsupervised.pth')
'''Epoch [1/10], Loss: 1.3973
Epoch [2/10], Loss: 1.3950
Epoch [3/10], Loss: 1.3940
Epoch [4/10], Loss: 1.3940
Epoch [5/10], Loss: 1.3933
Epoch [6/10], Loss: 1.3936
Epoch [7/10], Loss: 1.3942
Epoch [8/10], Loss: 1.3934
Epoch [9/10], Loss: 1.3933
Epoch [10/10], Loss: 1.3931'''