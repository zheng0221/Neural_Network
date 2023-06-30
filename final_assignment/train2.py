import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

# 自定义自监督任务 - 投影头部网络
class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_size, output_size):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, features):
        return self.layers(features)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

# 移除全连接层
model.fc = nn.Identity()

# 构建投影头部网络
projection_head = ProjectionHead(num_ftrs, hidden_size=256, output_size=num_ftrs)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(list(model.parameters()) + list(projection_head.parameters()), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
projection_head.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, _ in train_dataloader:
        images = images.to(device)

        # 提取特征
        features = model(images)
        features = features.view(features.size(0), -1)  # 将特征展平
        
        # 计算低维表示
        projections = projection_head(features)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss = criterion(projections, features)  # 使用原始特征作为目标计算损失
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


# 保存训练好的模型
torch.save(model.state_dict(), 'resnet18_selfsupervised.pth')