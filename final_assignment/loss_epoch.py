import matplotlib.pyplot as plt

# 记录每个epoch的损失值
loss_values = [1.8212,1.4769,1.3065,1.1991,1.1001,1.0215,0.9541,0.8947,0.8402,0.7986]
num_epochs = 10
# 绘制折线图
plt.plot(range(1, num_epochs + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
