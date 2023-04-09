# this file is for test readfile and other tests
import struct
import numpy as np
from matplotlib import pyplot as plt
# 读取原始数据并进行预处理
def data_fetch_preprocessing():
    train_image = open('train-images.idx3-ubyte', 'rb')
    test_image = open('t10k-images.idx3-ubyte', 'rb')
    train_label = open('train-labels.idx1-ubyte', 'rb')
    test_label = open('t10k-labels.idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',
                             train_label.read(8))# 读取文件的前8字节,读入magic number和numbers of images
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,
                                         dtype=np.uint8), ndmin=1)# 读取60000个标签数据,ndmin指定数据的维度
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99# 将结果转化为10维的列向量[0.01,0.99,...,0.01]

    # 测试数据的标签
    magic_t, n_t = struct.unpack('>II',
                                 test_label.read(8))
    # y_test = np.fromfile(test_label,dtype=np.uint8).reshape(10000, 1)
    y_test_label = np.array(np.fromfile(test_label,
                         dtype=np.uint8),ndmin=1)
    y_test = np.ones((10,10000)) * 0.01
    for i in range(10000):
        y_test[y_test_label[i]][i] = 0.99
    # print(y_test[2])
    # 训练数据共有60000个
    # print(len(labels))
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T #.T表示转置，得到784*60000矩阵

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test_label), 784).T
    print(x_train.shape)
    # 可以通过这个函数观察图像
    data=x_train[:,0].reshape(28,28)
    plt.imshow(data,cmap='Greys',interpolation=None)
    plt.show()
    x_train = x_train / 255 * 0.99 + 0.01 #归一化，有利于梯度下降收敛和速度
    x_test = x_test / 255 * 0.99 + 0.01 #归一化

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test
# data_fetch_preprocessing()
l = np.array([[0],[1],[2],[3]])
print(np.argmax(l[:,0]))
loss = 7935
print("loss:%f" % (loss/10000))
'''w1 = np.array([[1,2,3],[3,4,5]])
b1 = np.zeros((5, 1))
a = 0.1
print(w1)
for i in range(5):
    print(i)
file = open('model_params.txt', 'w',encoding='UTF-8')
file.write("w1:\n")
for i in range (2):
    file.write(str(w1[i])+'\n')
file.write("w2:\n")
for i in range (len(b1)):
    file.write(str(b1[i]))

file.write("learningrate:%f"%a)
file.close()'''