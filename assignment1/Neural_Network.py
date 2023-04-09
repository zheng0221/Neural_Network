import numpy as np
import matplotlib.pyplot as plt
import math
import struct

class Nerual_Network(object):
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, lam):
        """
        :param inputnodes: 输入层结点数
        :param hiddennodes: 隐藏层结点数
        :param outputnodes: 输出层结点数
        :param learningrate: 学习率
        """
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        self.lam = lam
        # 输入层与隐藏层权重矩阵初始化
        self.w1 = np.random.randn(self.hiddennodes, self.inputnodes) * 0.01
        # 隐藏层与输出层权重矩阵初始化
        self.w2 = np.random.randn(self.outputnodes, self.hiddennodes) * 0.01
        # 构建第一层常量矩阵200 by 1 matrix
        self.b1 = np.zeros((self.hiddennodes, 1))
        # 构建第二层常量矩阵 10 by 1 matrix
        self.b2 = np.zeros((self.outputnodes, 1))
        # 定义迭代次数
        self.epoch = 5

    # 激活函数
    def sigmoid(self, x):
        """
        :param x: 输入数据
        :return:返回sigmoid激活函数值
        """
        from scipy.special import expit
        return expit(x) #代替numpy中的exp函数，expit(x) = 1/(1+exp(-x))
        # return np.maximum(0,x)
    
    # loss交叉熵函数损失
    def loss_function(self, origin_label, fp_result):
        entropy =  np.sum(-origin_label * (np.log(fp_result)) - (1 - origin_label) * (np.log(1 - fp_result)))
        return entropy

    def forward_propagation(self, input_data, weight_matrix, b):
        """

        :param input_data: 输入数据
        :param weight_matrix: 权重矩阵
        :return: 激活函数后输出的活性值
        """
        z = np.add(np.dot(weight_matrix, input_data), b)
        return z, self.sigmoid(z)
    
    # 反向传播
    def back_propagation(self, a, z, da, weight_matrix, b):
        dz = da * (z * (1 - z))
        weight_matrix -= self.learningrate * np.dot(dz, a.T) / 60000
        b -= self.learningrate * np.sum(dz, axis=1, keepdims=True) / 60000
        da_n = np.dot(weight_matrix.T, da)
        return da_n
    
    def train(self, input_data, label_data):
        loss = []
        for item in range(self.epoch):
            print('第%d轮次开始执行' % item)
            l = 0
            for i in range(60000):
                # 前向传播
                z1, a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)# reshape(-1,1)转换成1列
                z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
                # 计算da[2]
                dz2 = a2 - label_data[:, i].reshape(-1, 1)
                l += self.loss_function(label_data[:, i].reshape(-1,1),a2) #记录cost
                dz1 = np.dot(self.w2.T, dz2) * a1 * (1.0 - a1) #梯度的计算
                # 反向传播过程SGD并考虑L2正则化
                self.w2 -= (self.learningrate * np.dot(dz2, a1.T) + self.learningrate * 2 * self.lam * self.w2 / 60000) #更新参数时加上L2正则化的梯度
                self.b2 -= self.learningrate * dz2

                self.w1 -= (self.learningrate * np.dot(dz1, (input_data[:, i].reshape(-1, 1)).T) + self.learningrate * 2 * self.lam * self.w1 / 60000) #更新参数时加上L2正则化的梯度
                self.b1 -= self.learningrate * dz1
            self.learningrate = self.learningrate*0.9 #学习率下降策略
            #损失函数需要加上正则化的损失
            loss.append(l/60000 + self.lam*(np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))/60000)
        return loss

    '''def train_vector(self, train_data, train_label):
        for item in range(self.epoch):
            print('正在执行第%d轮次' % item)
            # 前向传播
            z1, a1 = self.forward_propagation(train_data, self.w1, self.b1)
            z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
            dz2 = a2 - train_label
            dz1 = np.dot(self.w2.T, dz2) * a1 * (1 - a1)
            # 反向传播
            self.w2 -= self.learningrate * np.dot(dz2, a1.T) / 60000
            self.b2 -= self.learningrate * np.sum(dz2, axis=1, keepdims=True) / 60000
            self.w1 -= self.learningrate * np.dot(dz1, train_data.T) / 60000
            self.b1 -= self.learningrate * np.sum(dz1, axis=1, keepdims=True) / 60000'''


    def predict(self, input_data, label):
        precision = 0
        loss = 0
        for i in range(10000):
            z1, a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
            z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
            loss += self.loss_function(label[:, i].reshape(-1,1),a2)
            #print(a2)
            #print('模型预测值为:{0},\n实际值为{1}'.format(np.argmax(a2), label[i]))
            if np.argmax(a2) == np.argmax(label[:,i]):
                precision += 1
        # 显示测试的分类精度和loss
        print("准确率：%d" % (100 * precision / 10000) + "%")
        print("loss:%f" % (loss/10000))
    
    '''def predict_vector(self, input_data, label):
        z1, a1 = self.forward_propagation(input_data, self.w1, self.b1)
        z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
        precision=0
        for item in range(10000):
            if np.argmax(a2[:,item])==label[item]:
                precision+=1
        print('准确率：{0}%'.format(precision*100/10000))'''


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
    #y_test = np.fromfile(test_label,dtype=np.uint8).reshape(10000, 1)
    y_test_label = np.array(np.fromfile(test_label,dtype=np.uint8), ndmin=1)
    y_test = np.ones((10,10000)) * 0.01
    for i in range(10000):
        y_test[y_test_label[i]][i] = 0.99# 将结果转化为10维的列向量[0.01,0.99,...,0.01]
    # print(y_train[0])
    # 训练数据共有60000个
    # print(len(labels))
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T #.T表示转置，得到784*60000矩阵

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test_label), 784).T
    print(x_train.shape)
    # 可以通过这个函数观察图像
    # data=x_train[:,0].reshape(28,28)
    # plt.imshow(data,cmap='Greys',interpolation=None)
    # plt.show()
    x_train = x_train / 255 * 0.99 + 0.01 #归一化，有利于梯度下降收敛和速度
    x_test = x_test / 255 * 0.99 + 0.01 #归一化

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test
# data_fetch_preprocessing()

if __name__ == '__main__':
    # 输入层数据维度784，隐藏层200，输出层10
    dl = Nerual_Network(784, 200, 10, 0.2, 0.1)
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    # 循环训练方法
    # 可视化训练集loss曲线
    loss = dl.train(x_train, y_train)
    print(loss)
    plt.figure()
    plt.plot(np.arange(1,dl.epoch+1), loss, 'r')
    plt.xlabel('epochs')
    plt.ylabel('Cost')
    plt.title('Cost vs. Training Epoch')
    plt.savefig("loss_curve.jpg")
    plt.show()
    # 保存模型
    print("start writing...\n")
    file = open('model_params.txt', 'w',encoding='UTF-8')
    file.write("w1:\n")
    for i in range (len (dl.w1)):
        file.write(str(dl.w1[i])+'\n')
    file.write("w2:\n")
    for i in range (len (dl.w2)):
        file.write(str(dl.w2[i])+'\n')
    file.write("b1:\n")
    for i in range (len (dl.b1)):
        file.write(str(dl.b1[i]))
    file.write("\n")
    file.write("b2:\n")
    for i in range (len (dl.b2)):
        file.write(str(dl.b2[i]))
    file.write("\n")
    file.write("learningrate:%f\n" % dl.learningrate)
    file.write("lambda:%f\n" % dl.lam)
    file.write("epoch:%d\n" % dl.epoch)
    file.close()
    print("write success!\n")
    # 可视化每层网络w1、w2参数
    from PIL import Image
    image = dl.w1
    image = (image - image.min()) / (image.max() - image.min()) #to normalize
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image, 'L')
    image.save("neww1.jpg")
    image = dl.w2
    image = (image - image.min()) / (image.max() - image.min()) #to normalize
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image, 'L')
    image.save("neww2.jpg")
    # 矩阵元素有正有负，不做变化图像会失真
    '''im = Image.fromarray(np.uint8(dl.w1))
    im.save("w1.png")
    im = Image.fromarray(np.uint8(dl.w2))
    im.save("w2.png")'''

    # 预测模型
    dl.predict(x_test, y_test)

    # 向量化训练方法
    # dl.train_vector(x_train,y_train)
    # dl.predict_vector(x_test,y_test)